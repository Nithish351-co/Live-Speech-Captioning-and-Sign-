"""Machine translation via NLLB-200, served through CTranslate2.

CTranslate2 is already a dependency (it's faster-whisper's inference engine), so the
only new libraries are `transformers` (for the NLLB tokenizer) and `huggingface_hub`
(to fetch a pre-converted checkpoint). We deliberately use an already-converted
CTranslate2 checkpoint instead of running ct2-transformers-converter ourselves: the
converter needs PyTorch installed just to load the original weights once, and that's
an extra multi-GB install that can fight with the pinned CUDA/cuDNN wheels already set
up for faster-whisper. Downloading a pre-converted repo needs no PyTorch at all.

First call downloads ~2.3 GB to ./models/ (one-time) and can take a few minutes.
"""
import os

import ctranslate2
import transformers
from huggingface_hub import snapshot_download

# Pre-converted checkpoint of facebook/nllb-200-distilled-600M. 600M is the practical
# size for a per-chunk live pipeline; distilled-1.3B is meaningfully better quality at
# roughly 2x the decode latency if the paper wants that trade-off point.
MT_REPO = os.environ.get("ARSIGN_MT_REPO", "entai2965/nllb-200-distilled-600M-ctranslate2")
MT_DIR = os.environ.get(
    "ARSIGN_MT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", os.path.basename(MT_REPO)),
)
MT_DEVICE = os.environ.get("ARSIGN_MT_DEVICE", "auto")  # auto | cuda | cpu
# Force cpu here if the ASR model + NLLB together don't fit on your GPU's VRAM.

# App's short language code -> FLORES-200 code NLLB expects.
# https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
FLORES = {
    "en": "eng_Latn", "hi": "hin_Deva", "ta": "tam_Taml", "te": "tel_Telu",
    "bn": "ben_Beng", "ar": "arb_Arab", "zh": "zho_Hans", "ja": "jpn_Jpan",
    "fr": "fra_Latn", "es": "spa_Latn",
}
SUPPORTED = set(FLORES)

_translator = None
_tokenizers: dict[str, "transformers.PreTrainedTokenizerBase"] = {}
_mt_failed = False


def _load():
    global _translator, _mt_failed
    if _translator is not None or _mt_failed:
        return

    try:
        if not os.path.isdir(MT_DIR):
            print(f"[MT] downloading '{MT_REPO}' to {MT_DIR} (first run only, ~2.3 GB)...")
            snapshot_download(MT_REPO, local_dir=MT_DIR)
    except Exception as e:  # noqa: BLE001 - no internet, blocked host, disk full, etc.
        print(f"[MT] could not fetch '{MT_REPO}': {e}")
        _mt_failed = True
        return

    if MT_DEVICE == "auto":
        candidates = [("cuda", "int8_float16"), ("cpu", "int8")]
    else:
        candidates = [(MT_DEVICE, "int8_float16" if MT_DEVICE == "cuda" else "int8")]

    for device, compute in candidates:
        try:
            _translator = ctranslate2.Translator(MT_DIR, device=device, compute_type=compute)
            print(f"MT model '{MT_REPO}' ready on {device}/{compute}")
            return
        except Exception as e:  # noqa: BLE001
            print(f"[MT {device}/{compute}] failed: {e}")

    _mt_failed = True
    print("[MT] model unavailable: captions/sign will pass through untranslated.")


def _tokenizer(src_flores: str):
    tok = _tokenizers.get(src_flores)
    if tok is None:
        tok = transformers.AutoTokenizer.from_pretrained(MT_DIR, src_lang=src_flores)
        _tokenizers[src_flores] = tok
    return tok


def is_ready() -> bool:
    return _translator is not None


def warm_up():
    """Call once at server startup so the first live chunk isn't the download+load."""
    _load()
    if is_ready():
        translate("hello", "en", "fr")


def translate(text: str, src: str, tgt: str) -> str:
    """src/tgt are the app's short codes (en, hi, ta, ...), not FLORES codes.

    Always returns a string; never raises. Falls back to the input text unchanged if
    the model isn't available or the language pair isn't mapped.
    """
    text = text.strip()
    if not text or src == tgt:
        return text
    src_f, tgt_f = FLORES.get(src), FLORES.get(tgt)
    if not src_f or not tgt_f:
        return text  # unmapped language: caller surfaces this via config_note

    _load()
    if not is_ready():
        return text

    try:
        tok = _tokenizer(src_f)
        source = tok.convert_ids_to_tokens(tok.encode(text))
        result = _translator.translate_batch(
            [source], target_prefix=[[tgt_f]], beam_size=1, max_decoding_length=128,
        )
        target = result[0].hypotheses[0][1:]  # drop the leading target-language token
        return tok.decode(tok.convert_tokens_to_ids(target))
    except Exception as e:  # noqa: BLE001 - never let a bad chunk kill the websocket
        print(f"[MT] translate('{src}'->'{tgt}') failed: {e}")
        return text