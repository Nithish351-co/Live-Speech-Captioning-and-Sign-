"""Step 4: ASR -> parallel text branch (MT) + sign branch (gloss, English-pivoted) -> JSON.

ASR backend (env ARSIGN_ASR_BACKEND):
  groq    (default)  hosted Whisper large-v3 on Groq (free tier, OpenAI-compatible API)
  openai             hosted whisper-1 on OpenAI (paid)
  local              on-device faster-whisper (ARSIGN_MODEL, default 'small'), sliding-window streaming

Hosted backends transcribe whole UTTERANCES: the server detects pauses in the microphone audio,
sends one complete phrase per request (no words cut in half, one request per phrase, so Groq's
free-tier limit of 20 requests/min is not a problem) and shows the caption right after the pause.

If a hosted backend fails (no internet, rate limit, bad key) and ARSIGN_LOCAL_FALLBACK=1 (default),
the local model is used instead (loaded on first use) and the API is retried after 30 s.

Settings (environment variables, or a .env file next to this file):
  GROQ_API_KEY / OPENAI_API_KEY   key for the chosen backend (WHISPER_API_KEY works for either)
  WHISPER_API_URL, WHISPER_MODEL  optional overrides of the backend's URL / model
  ARSIGN_MODEL                    local model size: base | small | medium (default small)
  ARSIGN_PAUSE_S                  silence that ends a phrase (default 0.5)
  ARSIGN_MAX_UTT_S                longest phrase before a forced split (default 5)
  ARSIGN_VAD_MIN                  minimum loudness (RMS) counted as speech (default 0.01)
  ARSIGN_DEBUG_WAV=1              save every phrase sent to the ASR in ./debug_audio/ so you can listen to it
"""
import asyncio
import glob
import io
import json
import os
import site
import sys
import threading
import time
import wave

# Force line-buffered/unbuffered stdout so every print() below shows up in the
# terminal immediately (some setups buffer output, which looks like "nothing printed").
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:  # noqa: BLE001
    pass

# Optional: load keys etc. from a .env file (pip install python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

# Make pip-installed CUDA DLLs discoverable on Windows (needed by the local backend and NLLB)
for sp in site.getsitepackages():
    for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
        os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]

import numpy as np
import requests
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from asr_stream import StreamingASR
from gloss import load_lexicon, text_to_gloss
from translate import translate as mt_translate, warm_up as mt_warm_up, is_ready as mt_ready, SUPPORTED as MT_LANGS

SR = 16000

# ---------------------------------------------------------------- ASR configuration
PRESETS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/audio/transcriptions",
        "model": "whisper-large-v3",          # or whisper-large-v3-turbo (faster, slightly less accurate)
        "key_env": "GROQ_API_KEY",
    },
    "openai": {
        "url": "https://api.openai.com/v1/audio/transcriptions",
        "model": "whisper-1",
        "key_env": "OPENAI_API_KEY",
    },
}
ASR_BACKEND = os.environ.get("ARSIGN_ASR_BACKEND", "groq").lower()
LOCAL_FALLBACK = os.environ.get("ARSIGN_LOCAL_FALLBACK", "1") != "0"

API_URL = API_MODEL = API_KEY = ""
if ASR_BACKEND in PRESETS:
    _p = PRESETS[ASR_BACKEND]
    API_URL = os.environ.get("WHISPER_API_URL", _p["url"])
    API_MODEL = os.environ.get("WHISPER_MODEL", _p["model"])
    API_KEY = os.environ.get(_p["key_env"]) or os.environ.get("WHISPER_API_KEY", "")
    if not API_KEY:
        print(f"[ASR] {_p['key_env']} is not set -> using the LOCAL model instead.\n"
              f"      Put {_p['key_env']}=... in a .env file next to server.py.")
        ASR_BACKEND = "local"
elif ASR_BACKEND != "local":
    raise SystemExit(f"Unknown ARSIGN_ASR_BACKEND '{ASR_BACKEND}' (use groq, openai or local)")

API_TIMEOUT_S = 30
API_COOLDOWN_S = 30      # after a failed API call, use the local model for this long
SILENCE_PEAK = 0.005     # audio quieter than this is never sent
PAUSE_S = float(os.environ.get("ARSIGN_PAUSE_S", "0.5"))
MAX_UTT_S = float(os.environ.get("ARSIGN_MAX_UTT_S", "5"))
VAD_MIN = float(os.environ.get("ARSIGN_VAD_MIN", "0.01"))
DEBUG_WAV = os.environ.get("ARSIGN_DEBUG_WAV") == "1"

MODEL_NAME = os.environ.get("ARSIGN_MODEL", "small")
ENGLISH_ONLY_MODEL = MODEL_NAME.endswith(".en")

# Hosted Whisper reports the language as an English name; the rest of the app uses short codes.
LANG_CODES = {
    "english": "en", "hindi": "hi", "tamil": "ta", "telugu": "te", "bengali": "bn",
    "arabic": "ar", "chinese": "zh", "japanese": "ja", "french": "fr", "spanish": "es",
}

LEXICON = load_lexicon()

# ---------------------------------------------------------------- local model (primary or backup)
_local_model = None
_local_lock = threading.Lock()


def get_local_model():
    """Loads faster-whisper once (lazily when it is only the backup)."""
    global _local_model
    with _local_lock:
        if _local_model is not None:
            return _local_model
        from faster_whisper import WhisperModel
        for device, compute in (("cuda", "float16"), ("cpu", "int8")):
            try:
                m = WhisperModel(MODEL_NAME, device=device, compute_type=compute)
                list(m.transcribe(np.zeros(32000, dtype=np.float32), beam_size=1)[0])  # warm-up
                print(f"Local ASR model '{MODEL_NAME}' ready on {device}/{compute}")
                _local_model = m
                return m
            except Exception as e:  # noqa: BLE001
                print(f"[local {device}/{compute}] failed: {e}")
        raise RuntimeError("No usable device for the local ASR model")


def transcribe_words_local(audio: np.ndarray, lang: str):
    """-> ([(word, start_s, end_s)], detected_language_code); used by the sliding-window StreamingASR."""
    model = get_local_model()
    kw = {}
    if ENGLISH_ONLY_MODEL:
        kw["language"] = "en"
    elif lang and lang != "auto":
        kw["language"] = lang
    seconds = min(len(audio) / SR, 30.0)
    segments, info = model.transcribe(
        audio, beam_size=1, vad_filter=False, word_timestamps=True,
        temperature=0.0, condition_on_previous_text=False,
        max_new_tokens=min(int(seconds * 10) + 10, 440), **kw)
    words = [
        (word.word.strip(), word.start, word.end)
        for segment in segments
        for word in (segment.words or [])
        if word.word.strip()
    ]
    return words, info.language


def transcribe_text_local(audio: np.ndarray, lang: str):
    words, detected = transcribe_words_local(audio, lang)
    return " ".join(w for w, _, _ in words), detected


# ---------------------------------------------------------------- hosted Whisper API (Groq / OpenAI)
def _wav_bytes(audio: np.ndarray) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def transcribe_text_api(audio: np.ndarray, lang: str):
    """Hosted Whisper -> (text, detected_language_code)."""
    if len(audio) < 1600 or float(np.abs(audio).max()) < SILENCE_PEAK:
        return "", ""   # < 0.1 s is rejected by the API; silence is not worth a request

    data = {"model": API_MODEL, "response_format": "verbose_json", "temperature": "0"}
    if lang and lang != "auto":
        data["language"] = lang
    wav = _wav_bytes(audio)

    resp = None
    for attempt in range(3):
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
                files={"file": ("utterance.wav", wav, "audio/wav")},
                data=data,
                timeout=API_TIMEOUT_S,
            )
        except requests.RequestException as e:
            if attempt == 2:
                raise RuntimeError(f"Unable to reach the Whisper API: {e}") from e
            time.sleep(0.4 * (attempt + 1))
            continue
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            time.sleep(0.4 * (attempt + 1))
            continue
        break

    if resp is None or not resp.ok:
        detail = resp.text[:300] if resp is not None else "no response"
        raise RuntimeError(f"Whisper API error {getattr(resp, 'status_code', '?')}: {detail}")

    result = resp.json()
    segments = result.get("segments") or []
    if segments:
        # drop segments Whisper itself thinks are not speech (the classic "Thank you." on silence)
        text = " ".join(
            s.get("text", "").strip() for s in segments
            if not (s.get("no_speech_prob", 0.0) > 0.6 and s.get("avg_logprob", 0.0) < -1.0)
        ).strip()
    else:
        text = str(result.get("text", "")).strip()
    detected = str(result.get("language") or "").lower()
    return text, LANG_CODES.get(detected, detected)


def check_api_key() -> None:
    """Startup check so a wrong key is obvious immediately, not after you start speaking."""
    try:
        base = API_URL.rsplit("/audio/", 1)[0]
        r = requests.get(base + "/models", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=8)
        if r.ok:
            print(f"[ASR] {ASR_BACKEND} API key accepted.")
        else:
            print(f"[ASR] WARNING: {ASR_BACKEND} API key check failed (HTTP {r.status_code}); "
                  f"the local model will be used. Check the key in your .env file.")
    except requests.RequestException as e:
        print(f"[ASR] WARNING: could not reach the {ASR_BACKEND} API ({e}); the local model will be used.")


_api_blocked_until = 0.0


def transcribe_text(audio: np.ndarray, lang: str):
    """Decode function for UtteranceASR: hosted API first, local model as the backup."""
    global _api_blocked_until
    if DEBUG_WAV:
        os.makedirs("debug_audio", exist_ok=True)
        with open(os.path.join("debug_audio", f"utt_{int(time.time() * 1000)}.wav"), "wb") as f:
            f.write(_wav_bytes(audio))
    used, t0 = ASR_BACKEND, time.perf_counter()
    result = None
    if ASR_BACKEND != "local" and time.time() >= _api_blocked_until:
        try:
            result = transcribe_text_api(audio, lang)
        except Exception as e:  # noqa: BLE001
            if not LOCAL_FALLBACK:
                raise
            _api_blocked_until = time.time() + API_COOLDOWN_S
            print(f"[ASR] {ASR_BACKEND} API failed ({e}); using the local model for {API_COOLDOWN_S}s")
    if result is None:
        used = "local"
        result = transcribe_text_local(audio, lang)
    print(f"[ASR] {used} | {len(audio) / SR:.1f}s audio | {(time.perf_counter() - t0) * 1000:.0f} ms | "
          f"lang={result[1]!r} | {result[0]!r}")
    return result


# ---------------------------------------------------------------- utterance segmentation
class UtteranceASR:
    """Pause-based segmentation: collect audio until the speaker pauses, then transcribe the whole
    phrase once. Same process()/flush() interface as StreamingASR."""
    FRAME = SR // 10   # 100 ms analysis frames

    def __init__(self, decode_text, pause_s=PAUSE_S, max_utt_s=MAX_UTT_S, soft_s=3.0,
                 min_speech_s=0.3, pre_roll_s=0.3):
        self._decode = decode_text
        self._pause = pause_s
        self._max = max_utt_s
        self._soft = soft_s
        self._min_speech = min_speech_s
        self._pre = int(pre_roll_s * SR)
        self._buf = np.empty(0, dtype=np.float32)
        self._floor = 0.02          # running estimate of the background-noise level (RMS)
        self._detected = ""
        self._calls = 0

    def _rms(self, x: np.ndarray) -> np.ndarray:
        n = len(x) // self.FRAME
        if n == 0:
            return np.empty(0, dtype=np.float32)
        return np.sqrt((x[:n * self.FRAME].reshape(n, self.FRAME) ** 2).mean(axis=1))

    def _threshold(self) -> float:
        return min(max(VAD_MIN, 2.0 * self._floor), 0.04)

    def _speech_mask(self, rms: np.ndarray) -> np.ndarray:
        return rms > self._threshold()

    def process(self, audio: np.ndarray, language: str):
        audio = audio.astype(np.float32, copy=False)
        for r in self._rms(audio):                      # minimum-statistics noise floor
            self._floor = r if r < self._floor else self._floor * 1.002
        self._floor = float(np.clip(self._floor, 0.001, 0.05))
        self._buf = np.concatenate((self._buf, audio))

        rms = self._rms(self._buf)
        speech = self._speech_mask(rms)
        self._calls += 1
        if self._calls % 3 == 0:                        # diagnostics: is audio arriving and loud enough?
            print(f"[vad] mic level={float(rms.max()) if len(rms) else 0.0:.4f} "
                  f"noise-floor={self._floor:.4f} threshold={self._threshold():.4f} "
                  f"-> {'SPEECH' if speech.any() else 'silence'}")
        if not speech.any():                            # only silence: keep a short pre-roll
            self._buf = self._buf[-self._pre:]
            return [], self._detected

        n_speech_s = speech.sum() / 10.0
        trailing_s = (len(speech) - 1 - np.nonzero(speech)[0][-1]) / 10.0
        dur = len(self._buf) / SR

        if trailing_s >= self._pause:                   # speaker paused: phrase is complete
            if n_speech_s < self._min_speech:           # a click or cough, not speech
                self._buf = self._buf[-self._pre:]
                return [], self._detected
            last = np.nonzero(speech)[0][-1]
            return self._emit(min(len(self._buf), (last + 1) * self.FRAME + int(0.3 * SR)), language)

        if dur >= self._soft:                           # long phrase: split at the latest short gap
            cut = self._gap_cut(speech)
            if cut is not None:
                return self._emit(cut * self.FRAME, language)
        if dur >= self._max:                            # no gap at all: split at the quietest moment
            lo = max(0, len(rms) - 15)
            return self._emit((lo + int(np.argmin(rms[lo:])) + 1) * self.FRAME, language)
        return [], self._detected

    @staticmethod
    def _gap_cut(speech: np.ndarray):
        """Middle of the latest silent run (>= 0.3 s) that starts after the first 2 s."""
        best, run = None, 0
        for i, s in enumerate(speech):
            if s:
                if run >= 3 and i - run >= 20:
                    best = i - run // 2
                run = 0
            else:
                run += 1
        return best

    def _emit(self, cut: int, language: str):
        utt, self._buf = self._buf[:cut], self._buf[cut:]
        return self._transcribe(utt, language)

    def _transcribe(self, utt: np.ndarray, language: str):
        lang = language
        if lang == "auto" and self._detected and len(utt) / SR < 2.0:
            lang = self._detected   # very short audio: language detection is unreliable, keep the last one
        text, detected = self._decode(utt, lang)
        if detected:
            self._detected = detected
        return text.split(), self._detected

    def flush(self, language: str):
        utt, self._buf = self._buf, np.empty(0, dtype=np.float32)
        if len(utt) < self.FRAME or not self._speech_mask(self._rms(utt)).any():
            return [], self._detected
        return self._transcribe(utt, language)


# ---------------------------------------------------------------- startup checks
if ASR_BACKEND == "local":
    get_local_model()
else:
    print(f"ASR backend: {ASR_BACKEND} API, model '{API_MODEL}', local fallback "
          f"{'on' if LOCAL_FALLBACK else 'off'}, pause={PAUSE_S:g}s, max phrase={MAX_UTT_S:g}s")
    check_api_key()

try:
    text_to_gloss("warm up")
except LookupError as e:
    raise SystemExit(f"NLTK data missing: run `python setup_nltk.py` first.\n{e}")

# Loads (or downloads, first run only) the NLLB model. Never raises: if it fails
# (offline, no VRAM, etc.) MT falls back to pass-through and config_note() says so.
mt_warm_up()

app = FastAPI()


def text_branch(text: str, src: str, tgt: str):
    t0 = time.perf_counter()
    caption = mt_translate(text, src, tgt)
    return caption, (time.perf_counter() - t0) * 1000


def sign_branch(text: str, lang: str):
    t0 = time.perf_counter()
    # The gloss rules operate on English; non-English speech is pivoted through NLLB
    # first. This is a second MT call on top of the caption's translate(), so the
    # parallelism ablation should treat "gloss on non-English input" as its own case.
    en_text = text if lang == "en" else mt_translate(text, lang, "en")
    gloss = text_to_gloss(en_text)
    # Cue builder: a gloss word either hits the lexicon (plays a real clip) or falls
    # back to fingerspelling (letter-by-letter, no clip). Nothing is silently dropped,
    # which matters for the lexicon-coverage metric the paper reports.
    cues = []
    for g in gloss:
        clip = LEXICON.get(g)
        cues.append({
            "gloss": g,
            "clip": clip,
            "fingerspell": None if clip else [c for c in g if c.isalpha()],
        })
    hits = sum(1 for c in cues if c["clip"])
    fingerspell_n = sum(1 for c in cues if c["fingerspell"])
    return gloss, cues, hits, fingerspell_n, (time.perf_counter() - t0) * 1000


def config_note(src: str, tgt: str) -> str:
    notes = []
    if ASR_BACKEND == "local" and ENGLISH_ONLY_MODEL and src not in ("en", "auto"):
        notes.append("ASR model is English-only (set ARSIGN_MODEL=base or small for multilingual speech).")
    if not mt_ready():
        notes.append("MT model unavailable (offline or failed to load): captions and sign use the recognized text as-is.")
    else:
        if src != "auto" and src not in MT_LANGS:
            notes.append(f"No MT mapping for source language '{src}': recognized text is used as-is.")
        if tgt not in MT_LANGS:
            notes.append(f"No MT mapping for caption language '{tgt}': captions show the recognized text untranslated.")
    return " ".join(notes)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("[ws] client connected")
    loop = asyncio.get_running_loop()
    cfg = {"source_lang": "auto", "target_lang": "en"}
    if ASR_BACKEND == "local":
        streaming_asr = StreamingASR(transcribe_words_local)   # sliding window, on-device
    else:
        streaming_asr = UtteranceASR(transcribe_text)          # pause-based phrases, hosted API
    caption_words: list[str] = []

    async def publish_words(words: list[str], detected: str, chunk_id: int, asr_ms: float):
        committed_text = " ".join(words)
        caption_words.extend(words)
        text = " ".join(caption_words)
        src = detected if cfg["source_lang"] == "auto" else cfg["source_lang"]
        if committed_text:
            (caption, text_ms), (gloss, cues, hits, fingerspell_n, sign_ms) = await asyncio.gather(
                loop.run_in_executor(None, text_branch, text, src, cfg["target_lang"]),
                loop.run_in_executor(None, sign_branch, committed_text, src),
            )
            if text.rstrip().endswith((".", "!", "?", "।", "۔", "。", "！", "？", "؟")):
                caption_words.clear()
        else:
            caption, text_ms = "", 0.0
            gloss, cues, hits, fingerspell_n, sign_ms = [], [], 0, 0, 0.0

        payload = {
            "id": chunk_id, "text": text, "lang": src, "caption": caption,
            "gloss": gloss, "cues": cues, "gloss_n": len(gloss),
            "lex_hits": hits, "fingerspell_n": fingerspell_n,
            "asr_ms": round(asr_ms, 1), "text_ms": round(text_ms, 2), "sign_ms": round(sign_ms, 2),
        }
        await ws.send_text(json.dumps(payload, ensure_ascii=False))

    while True:
        msg = await ws.receive()
        if msg["type"] == "websocket.disconnect":
            print("[ws] client disconnected")
            break

        if msg.get("text") is not None:  # config message
            data = json.loads(msg["text"])
            if data.get("type") == "flush":
                try:
                    final_words, detected = await loop.run_in_executor(
                        None, streaming_asr.flush, cfg["source_lang"])
                    if final_words:
                        await publish_words(final_words, detected, -1, 0.0)
                except Exception as e:  # noqa: BLE001
                    print(f"[ws] final ASR flush failed: {e!r}")
                    await ws.send_text(json.dumps({
                        "id": -1, "type": "chunk_error", "error": str(e),
                    }))
                await ws.send_text(json.dumps({"type": "flush_ack"}))
                continue
            if data.get("type") == "config":
                for k in ("source_lang", "target_lang"):
                    if k in data:
                        cfg[k] = data[k]
                print(f"[ws] config: source={cfg['source_lang']} target={cfg['target_lang']}")
                await ws.send_text(json.dumps({
                    "type": "config_ack", "mt": mt_ready(),
                    "note": config_note(cfg["source_lang"], cfg["target_lang"]),
                }))
            continue

        raw = msg.get("bytes")
        if not raw:
            print(f"[ws] received a text/empty websocket frame with no 'bytes' payload, skipping: {msg}")
            continue
        # Protocol: 4-byte little-endian chunk id + int16 PCM @ 16 kHz mono
        chunk_id = int.from_bytes(raw[:4], "little")

        # One bad chunk must never be able to take the whole connection down.
        # Everything from decode through send is guarded: on failure we log,
        # tell the client this chunk was skipped, and go back to `receive()`
        # so the pipeline keeps running for every chunk after it.
        try:
            audio = np.frombuffer(raw[4:], dtype=np.int16).astype(np.float32) / 32768.0
            t0 = time.perf_counter()
            committed_words, detected = await loop.run_in_executor(
                None, streaming_asr.process, audio, cfg["source_lang"])
            asr_ms = (time.perf_counter() - t0) * 1000
            await publish_words(committed_words, detected, chunk_id, asr_ms)
        except Exception as e:  # noqa: BLE001
            print(f"[ws] chunk {chunk_id} FAILED, skipping: {e!r}")
            import traceback
            traceback.print_exc()
            try:
                await ws.send_text(json.dumps({"id": chunk_id, "type": "chunk_error", "error": str(e)}))
            except Exception:  # noqa: BLE001
                print("[ws] socket itself is dead, ending connection loop")
                break


# Serve the frontend (must be mounted after the websocket route)
@app.get("/", include_in_schema=False)
async def phone_ui():
    """Serve the phone-style AR sign captioning interface by default."""
    return FileResponse(
        os.path.join("static", "phone.html"),
        headers={"Cache-Control": "no-store"},
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")