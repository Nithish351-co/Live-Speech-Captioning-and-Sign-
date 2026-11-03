"""Offline latency benchmark for the ASR -> (MT) -> text/sign branches pipeline.

Feeds one recorded WAV through the same stages the live server runs, so results are
reproducible (no mic, no browser). Run it once per configuration you want to compare
(model size, device, languages) and let it sweep chunk length and branch mode:

    python bench.py --wav speech.wav --label gtx1650-base.en --model base.en --device cuda
    python bench.py --wav speech.wav --label cpu-base.en    --model base.en --device cpu
    python bench.py --wav hindi.wav  --label pivot-hi-en    --model small --src hi --tgt hi

Each run appends per-chunk rows to bench_results.csv and prints a summary table.

Real-time simulation: audio arrives in real time (chunk i is complete at (i+1)*chunk_s) and
the server handles chunks one at a time, so a slow chunk delays the ones behind it. The
e2e_ms column is the delay AFTER the chunk's audio ended, including that queueing. A word
spoken at the start of a chunk waits an extra chunk_s on top of this.
"""
import argparse
import csv
import glob
import os
import site
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SR = 16000


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        n_ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 2:
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 1:  # 8-bit WAV is unsigned, centred on 128
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        sys.exit("WAV must be 8- or 16-bit PCM (convert: ffmpeg -i in.wav -ar 16000 -ac 1 out.wav)")
    if n_ch > 1:
        a = a.reshape(-1, n_ch).mean(axis=1)
    if rate < SR:
        print(f"note: input is {rate} Hz; upsampling to {SR} Hz cannot restore the missing "
              f"high frequencies, so ASR accuracy may be lower than with native 16 kHz audio")
    if rate != SR:
        n_new = int(len(a) * SR / rate)
        a = np.interp(np.linspace(0, 1, n_new, endpoint=False),
                      np.linspace(0, 1, len(a), endpoint=False), a).astype(np.float32)
    return a


def decode_kwargs(decode: str, n_samples: int) -> dict:
    """Extra faster-whisper arguments. 'default' keeps the library's temperature-fallback
    (a failed decode is retried up to 5 more times at higher temperature: slow and bursty).
    'bounded' decodes once, greedily, with a token cap that scales with chunk length."""
    if decode == "bounded":
        # Whisper decodes long audio in 30 s windows and its decoder holds at most 448 tokens
        # (prompt included), so budget for one window and stay under that hard limit.
        # For audio up to 30 s this equals the earlier formula, so existing results are unchanged.
        secs = min(n_samples / SR, 30.0)
        return {"temperature": 0.0, "condition_on_previous_text": False,
                "max_new_tokens": min(int(secs * 10) + 10, 440)}
    return {}


def make_chunks(audio: np.ndarray, chunk_s: float) -> list:
    cs = int(chunk_s * SR)
    return [audio[i:i + cs] for i in range(0, len(audio) - cs + 1, cs)]


def simulate_queue(totals_ms: list, chunk_s: float) -> list:
    """Per-chunk delay after the chunk's audio ended, with one-at-a-time processing."""
    finish, out = 0.0, []
    for i, total in enumerate(totals_ms):
        arrival = (i + 1) * chunk_s * 1000
        start = max(arrival, finish)
        finish = start + total
        out.append(finish - arrival)
    return out


def run_condition(chunks, chunk_s, mode, asr_fn, text_fn, sign_fn):
    """asr_fn(audio)->text | text_fn(text)->(caption, ms) | sign_fn(text)->(n, hits, ms)"""
    rows, totals = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, audio in enumerate(chunks):
            t0 = time.perf_counter()
            text = asr_fn(audio)
            asr_ms = (time.perf_counter() - t0) * 1000

            tb = time.perf_counter()
            if mode == "parallel":
                f_text, f_sign = pool.submit(text_fn, text), pool.submit(sign_fn, text)
                (caption, text_ms), (gloss_n, hits, sign_ms) = f_text.result(), f_sign.result()
            else:
                caption, text_ms = text_fn(text)
                gloss_n, hits, sign_ms = sign_fn(text)
            branch_ms = (time.perf_counter() - tb) * 1000

            total = asr_ms + branch_ms
            totals.append(total)
            rows.append({"chunk": i, "asr_ms": asr_ms, "text_ms": text_ms, "sign_ms": sign_ms,
                         "branch_wall_ms": branch_ms, "total_ms": total,
                         "text_len": len(text), "gloss_n": gloss_n, "lex_hits": hits,
                         "sample_text": text})
    for r, e in zip(rows, simulate_queue(totals, chunk_s)):
        r["e2e_ms"] = e
    return rows


def summarize(rows):
    def col(k):
        return np.array([r[k] for r in rows], dtype=float)
    total = col("total_ms")
    return {"n": len(rows), "asr_med": np.median(col("asr_ms")),
            "branch_med": np.median(col("branch_wall_ms")),
            "total_med": np.median(total), "total_p95": np.percentile(total, 95),
            "e2e_med": np.median(col("e2e_ms")), "e2e_p95": np.percentile(col("e2e_ms"), 95),
            "e2e_max": col("e2e_ms").max(), "mean_total": total.mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--model", default=os.environ.get("ARSIGN_MODEL", "base.en"))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--mt-device", default=None, help="cuda|cpu for translation (default: auto)")
    ap.add_argument("--src", default="en")
    ap.add_argument("--tgt", default="en")
    ap.add_argument("--chunks", type=float, nargs="+", default=[2.0])
    ap.add_argument("--modes", nargs="+", default=["parallel", "sequential"])
    ap.add_argument("--out", default="bench_results.csv")
    ap.add_argument("--decode", default="default", choices=["default", "bounded"],
                    help="bounded = no temperature fallback + token cap (removes latency spikes)")
    args = ap.parse_args()
    if args.decode == "bounded":
        args.label += "-bounded"   # keeps the CSV schema unchanged; the label carries the setting

    if args.mt_device:
        os.environ["ARSIGN_MT_DEVICE"] = args.mt_device
    for sp in site.getsitepackages():  # Windows: pip-installed CUDA DLLs
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]

    from faster_whisper import WhisperModel
    from gloss import load_lexicon, text_to_gloss
    import translate as mt

    compute = "float16" if args.device == "cuda" else "int8"
    model = WhisperModel(args.model, device=args.device, compute_type=compute)
    english_only = args.model.endswith(".en")
    needs_mt = args.src != args.tgt or args.src != "en"
    if needs_mt:
        mt.warm_up()
        if not mt.is_ready():
            sys.exit("MT model not available; fix that first or use --src en --tgt en")
    lexicon = load_lexicon()

    def asr_fn(audio):
        kw = {"language": "en"} if english_only else ({"language": args.src} if args.src != "auto" else {})
        segs, _ = model.transcribe(audio, beam_size=1, vad_filter=True,
                                   **decode_kwargs(args.decode, len(audio)), **kw)
        return " ".join(s.text.strip() for s in segs).strip()

    def text_fn(text):
        t = time.perf_counter()
        return mt.translate(text, args.src, args.tgt), (time.perf_counter() - t) * 1000

    def sign_fn(text):
        t = time.perf_counter()
        en = text if args.src == "en" else mt.translate(text, args.src, "en")
        gloss = text_to_gloss(en)
        return len(gloss), sum(1 for g in gloss if g in lexicon), (time.perf_counter() - t) * 1000

    audio = load_wav(args.wav)
    print(f"{args.wav}: {len(audio) / SR:.1f}s of audio | model={args.model} device={args.device} "
          f"src={args.src} tgt={args.tgt}")

    # Discard one full pass so model/GPU warm-up doesn't pollute the numbers
    warm = make_chunks(audio, min(args.chunks))[:1]
    if warm:
        run_condition(warm, min(args.chunks), "sequential", asr_fn, text_fn, sign_fn)

    write_header = not os.path.exists(args.out)
    print(f"\n{'chunk_s':<8}{'mode':<11}{'n':<4}{'asr':<8}{'branch':<8}{'total':<8}{'p95':<8}"
          f"{'e2e_med':<9}{'e2e_p95':<9}{'e2e_max':<9}RTF")
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = None
        for cs in args.chunks:
            chunks = make_chunks(audio, cs)
            if not chunks:
                print(f"{cs}: audio shorter than one chunk, skipped")
                continue
            for mode in args.modes:
                rows = run_condition(chunks, cs, mode, asr_fn, text_fn, sign_fn)
                s = summarize(rows)
                rtf = s["mean_total"] / (cs * 1000)
                print(f"{cs:<8}{mode:<11}{s['n']:<4}{s['asr_med']:<8.0f}{s['branch_med']:<8.0f}"
                      f"{s['total_med']:<8.0f}{s['total_p95']:<8.0f}{s['e2e_med']:<9.0f}"
                      f"{s['e2e_p95']:<9.0f}{s['e2e_max']:<9.0f}{rtf:.2f}")
                for r in rows:
                    r.update({"label": args.label, "model": args.model, "device": args.device,
                              "src": args.src, "tgt": args.tgt, "chunk_s": cs, "mode": mode})
                    r = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}
                    if w is None:
                        w = csv.DictWriter(f, fieldnames=list(r.keys()))
                        if write_header:
                            w.writeheader()
                    w.writerow(r)
    print(f"\nRows appended to {args.out}. RTF > 1 means processing is slower than real time "
          f"(the backlog grows without bound).")


if __name__ == "__main__":
    main()