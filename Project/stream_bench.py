"""Sliding-window ("LocalAgreement") streaming decode vs. today's independent-chunk decode.

The former server path (and bench.py's chunk baseline) decoded each 1-3s audio chunk
independently. That's why segment_eval.py found fragments like "taking longer to." +
"places to save money.": each chunk was treated as a complete, isolated utterance, so
Whisper cut words at chunk boundaries and punctuated each partial segment. The live
server now uses the sliding-window LocalAgreement strategy measured by this script.

Sliding-window decoding avoids (a) by re-decoding a GROWING window of recent audio
instead of disjoint chunks, and uses the standard LocalAgreement-n policy (Macháček et
al. 2023, "Turning Whisper into Real-Time Transcription System" -- check the exact
citation before it goes in the paper) to avoid re-showing captions that keep changing: a
word is only committed (finalized, shown as a permanent caption) once it agrees across n
consecutive re-decodes of the growing window. Words near the window's trailing edge are
expected to be unstable -- more audio hasn't arrived yet to confirm them -- so they're
held back as a provisional tail instead of being committed immediately, the way
independent-chunk decoding effectively "commits" every word immediately with no chance
to revise a boundary mistake once made.

The default synthetic mode is offline-testable and has no GPU/network dependency:
decode_window() is a deterministic simulated decoder over a known script, for testing
the commit-policy logic rather than Whisper recognition quality. Pass --wav to run the
separate real-audio comparison on a local Whisper model. The live policy is implemented
in asr_stream.py; this benchmark is useful for comparing its design parameters but does
not exercise the app's WebSocket, microphone, translation, or sign-rendering paths.

    python stream_bench.py                    # runs the built-in synthetic script
    python stream_bench.py --show              # also print each mode's committed transcript
    python stream_bench.py --wpm 190 --chunk-s 1 2 --stride-s 0.5 1 2

"""
import argparse
import math
import re
import time
from dataclasses import dataclass

# ============================================================================
# Shared scoring: same token-level edit distance definition as eval_gloss.py
# ============================================================================

def levenshtein(a: list, b: list) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def wer(pred: list, ref: list) -> float:
    return levenshtein(pred, ref) / max(1, len(ref))


def pct(values: list, p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def median(values: list) -> float:
    return pct(values, 50)


# ============================================================================
# Synthetic ground truth: a script with known per-word start/end times, standing
# in for what real word-level ASR timestamps would give us. Themed the same as
# the real fragments segment_eval.py surfaced ("places to save money", "tax"),
# but this is still a placeholder script, not your real recording.
# ============================================================================

@dataclass
class Word:
    text: str
    start: float
    end: float


SIM_SCRIPT = (
    "Hello and welcome. Thanks for taking the time to listen today. "
    "I want to talk about a few places to save money without giving up "
    "things that matter. First, look at subscriptions you forgot about. "
    "Second, compare insurance rates once a year instead of renewing automatically. "
    "Third, cook at home more often, since eating out adds up fast. "
    "These small changes, done consistently, can save a family real money on tax day."
)


def build_words(text: str, wpm: float) -> list:
    """Assigns each word a [start, end) in seconds at a fixed speaking rate, with a
    short extra pause after sentence-final punctuation. Deterministic, no randomness,
    so re-running this script always reproduces the same case."""
    sec_per_word = 60.0 / wpm
    words, t = [], 0.0
    for tok in text.split():
        dur = sec_per_word * (0.75 + 0.25 * min(len(tok), 8) / 8)
        words.append(Word(tok, t, t + dur))
        t += dur
        if tok.endswith((".", "!", "?")):
            t += sec_per_word * 0.7
    return words


def _truncate(word: str, frac: float) -> str:
    core = word.rstrip(".,!?;:")
    return core[:max(1, round(len(core) * frac))]


# ============================================================================
# REAL-AUDIO: this is the one function to replace to run against real audio.
# It currently answers "what would ASR transcribe for the window [t_start, t_end)?"
# by looking up ground-truth words instead of calling a model. Swap the body for
# `model.transcribe(audio[int(t_start*SR):int(t_end*SR)], ...)`, splitting the
# result into (word_index, text) pairs isn't possible for a real model (no ground
# truth index), so the real version should return plain text and the two run_*
# functions below would need to switch from index-based agreement to word-string
# LocalAgreement -- see the module docstring.
# ============================================================================

def decode_window(words: list, t_start: float, t_end: float, edge_margin: float = 0.25) -> list:
    """Simulated ASR call on the window [t_start, t_end).

    Returns [(word_index, text), ...] for every word this window can "hear" fully and
    correctly (its end lands with edge_margin seconds of headroom before t_end), plus at
    most one further word straddling the trailing edge as a partial/garbled guess -- or
    dropped outright if too little of it has arrived. Words starting before t_start are
    skipped, on the assumption a previous window already emitted (or mangled) them --
    this only models trailing-edge instability, not front-edge stitching errors, which
    are a further real artifact but not the one this script is testing.
    """
    out = []
    for i, w in enumerate(words):
        if w.start < t_start or w.start >= t_end:
            continue
        if w.end <= t_end - edge_margin:
            out.append((i, w.text))
            continue
        frac_in = min(1.0, max(0.0, (t_end - w.start) / max(1e-6, w.end - w.start)))
        if frac_in >= 0.4:
            out.append((i, w.text if frac_in >= 0.9 else _truncate(w.text, frac_in)))
        break
    return out


# ============================================================================
# Mode 1: today's approach -- each chunk decoded alone, committed immediately,
# never revised.
# ============================================================================

def run_chunk_baseline(words: list, chunk_s: float) -> list:
    total = words[-1].end
    n = math.ceil(total / chunk_s)
    committed = []
    for k in range(n):
        t_start, t_end = k * chunk_s, (k + 1) * chunk_s
        tokens = decode_window(words, t_start, t_end)
        for idx, txt in tokens:
            committed.append({"idx": idx, "text": txt, "commit_t": t_end})
        if tokens and not committed[-1]["text"].rstrip().endswith((".", "!", "?")):
            # The artifact you actually saw: every independently-decoded chunk looks
            # complete to Whisper, so it gets sentence-final punctuation whether or
            # not this really was a sentence end.
            committed[-1]["text"] += "."
    return committed


# ============================================================================
# Mode 2: sliding window + LocalAgreement-n. Re-decode a growing window every
# stride_s; commit a word only once n consecutive hypotheses agree on it (same
# ground-truth index AND same text); trim the window's start to the last
# committed word so re-decode cost stays bounded.
# ============================================================================

def run_streaming(words: list, stride_s: float, max_window_s: float,
                   agreements: int = 2, edge_margin: float = 0.25) -> list:
    total = words[-1].end
    window_start = 0.0
    committed_upto = -1
    committed = []
    hyps: list = []
    t = 0.0
    while t < total:
        t = min(total, t + stride_s)
        hyp = decode_window(words, window_start, t, edge_margin)
        hyps.append(hyp)
        if len(hyps) > agreements:
            hyps.pop(0)

        if len(hyps) == agreements:
            agreed = 0
            while all(agreed < len(h) for h in hyps) and len({h[agreed] for h in hyps}) == 1:
                agreed += 1
            for idx, txt in hyps[-1][:agreed]:
                if idx > committed_upto:
                    committed.append({"idx": idx, "text": txt, "commit_t": t})
                    committed_upto = idx

        if committed_upto >= 0:
            window_start = words[committed_upto].end
        if t - window_start > max_window_s and hyps:
            # Safety valve: don't let one persistently-unstable tail block progress
            # forever. Commit the latest guess outright and start a fresh window.
            for idx, txt in hyps[-1]:
                if idx > committed_upto:
                    committed.append({"idx": idx, "text": txt, "commit_t": t, "forced": True})
                    committed_upto = idx
            window_start = words[committed_upto].end if committed_upto >= 0 else window_start
            hyps.clear()

    # End of audio: nothing more is coming to confirm or refute the trailing words,
    # so decode the remaining window one last time with no edge margin and commit
    # whatever's left outright.
    final = decode_window(words, window_start, total, edge_margin=0.0)
    for idx, txt in final:
        if idx > committed_upto:
            committed.append({"idx": idx, "text": txt, "commit_t": total, "forced": True})
            committed_upto = idx
    return committed


# ============================================================================
# Scoring + reporting
# ============================================================================

def summarize(committed: list, words: list) -> dict:
    pred = [c["text"] for c in committed]
    ref = [w.text for w in words]
    seen = {c["idx"] for c in committed}
    lat = [c["commit_t"] - words[c["idx"]].end for c in committed]
    spurious_punct = sum(
        1 for c in committed
        if c["text"].rstrip().endswith(".") and not words[c["idx"]].text.rstrip().endswith((".", "!", "?"))
    )
    return {
        "n_committed": len(committed), "dropped": len(words) - len(seen),
        "wer": wer(pred, ref), "spurious_punct": spurious_punct,
        "lat_med": median(lat), "lat_p95": pct(lat, 95), "lat_max": max(lat) if lat else 0.0,
    }


def transcript_of(committed: list) -> str:
    return " ".join(c["text"] for c in committed)


def print_table(rows: list) -> None:
    print(f"\n{'mode':<9}{'param':<8}{'n':<5}{'dropped':<9}{'wer':<7}{'spur.':<7}"
          f"{'lat_med':<9}{'lat_p95':<9}lat_max")
    for mode, param, s in rows:
        print(f"{mode:<9}{param:<8g}{s['n_committed']:<5}{s['dropped']:<9}{s['wer']:<7.3f}"
              f"{s['spurious_punct']:<7}{s['lat_med']:<9.2f}{s['lat_p95']:<9.2f}{s['lat_max']:.2f}")
    print("\nwer: word-level edit distance vs. the true script (lower is better).")
    print("spur.: how many committed words got an unwarranted sentence-final period glued on.")
    print("lat_*: seconds between a word actually ending and it being committed/shown (higher is slower-but-safer).")
    print("dropped: true words that never got committed at all (starved at a chunk boundary).")


def _norm_tokens(words: list) -> list:
    """Lower-case and strip punctuation, so 'Hello,' == 'hello'. Standard practice for WER; without
    it every stray comma/period from a chunk boundary counts as a word error."""
    out = []
    for w in words:
        t = re.sub(r"[^\w']+", "", w.lower())
        if t:
            out.append(t)
    return out


def _key(word: str) -> str:
    """Comparison key for LocalAgreement: 'Hello,' and 'hello' count as the same word."""
    return re.sub(r"[^\w']+", "", word.lower())


def align_counts(pred: list, ref: list) -> tuple:
    """(substitutions, deletions, insertions) of an optimal alignment. deletions = reference
    words missing from pred; insertions = extra words in pred. Sum equals levenshtein()."""
    n, m = len(pred), len(ref)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (pred[i - 1] != ref[j - 1]))
    i, j, s, dele, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (pred[i - 1] != ref[j - 1]):
            s += pred[i - 1] != ref[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ins += 1
            i -= 1
        else:
            dele += 1
            j -= 1
    return s, dele, ins


def run_real(args) -> None:
    """Real-audio comparison of independent chunks vs sliding-window LocalAgreement.

    Every mode reports: word errors against the single-pass offline decode (normalised, and raw
    for comparison with earlier runs), the substitution/deletion/insertion split, per-word
    latency, per-decode time and the real-time factor of the decoding loop.

    Latency of a word = (audio time at which it was shown) - (time it finished being spoken),
    where "shown" = end of the audio the decode had + the measured decode time. Queueing is not
    modelled (RTF > 1 means it would build up). Word times come from faster-whisper's
    word_timestamps, which adds some alignment cost, so decode times are slightly pessimistic
    compared with the live server.

    Streaming tracks words by absolute time and trims the window at the END OF THE LAST CONFIRMED
    WORD once it exceeds --max-window-s, so the unconfirmed tail is re-decoded in the next window
    instead of being lost. (Earlier versions hard-reset the window, dropping the tail, and then
    indexed words by position, which crashed when a re-decode returned fewer words.)
    """
    import glob
    import os
    import site
    import sys

    for sp in site.getsitepackages():  # Windows: pip-installed CUDA DLLs
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bench  # reuses load_wav + decode_kwargs
    from faster_whisper import WhisperModel

    compute = "float16" if args.device == "cuda" else "int8"
    model = WhisperModel(args.model, device=args.device, compute_type=compute)
    english_only = args.model.endswith(".en")
    kw = {} if english_only else ({"language": args.src} if args.src != "auto" else {})

    def transcribe_text(audio):
        if len(audio) == 0:
            return ""
        segs, _ = model.transcribe(audio, beam_size=1, vad_filter=False,
                                   **bench.decode_kwargs(args.decode, len(audio)), **kw)
        return " ".join(s.text.strip() for s in segs).strip()

    def transcribe_words(audio):
        """-> ([(word, start_s, end_s)] relative to this slice, decode_ms)"""
        if len(audio) == 0:
            return [], 0.0
        t0 = time.perf_counter()
        segs, _ = model.transcribe(audio, beam_size=1, vad_filter=False, word_timestamps=True,
                                   **bench.decode_kwargs(args.decode, len(audio)), **kw)
        words = [(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or []) if w.word.strip()]
        return words, (time.perf_counter() - t0) * 1000

    audio = bench.load_wav(args.wav)
    total_s = len(audio) / bench.SR
    print(f"{args.wav}: {total_s:.1f}s | model={args.model} device={args.device} decode={args.decode}")
    # warm-up so the first timed decode does not include CUDA/model initialisation
    transcribe_words(audio[:int(bench.SR)])
    print("Decoding the whole file once, offline -- this is the reference every mode below is scored against...")
    offline = transcribe_text(audio).split()
    if args.ref_text:
        with open(args.ref_text, encoding="utf-8") as f:
            reference = f.read().split()
        ref_label = "GROUND TRUTH from " + os.path.basename(args.ref_text)
    else:
        reference, ref_label = offline, "Whisper's own offline decode"
    ref_n = _norm_tokens(reference)
    print(f"reference ({len(reference)} words, {ref_label}): {' '.join(reference)}\n")

    stem = os.path.splitext(os.path.basename(args.wav))[0]

    def save(name, lines):
        """One line per unit the live server would gloss separately (a chunk, or the committed stream)."""
        if not args.save_transcripts:
            return
        os.makedirs(args.save_transcripts, exist_ok=True)
        with open(os.path.join(args.save_transcripts, f"{stem}__{name}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    save("offline", [" ".join(offline)])
    if args.ref_text:
        save("truth", [" ".join(reference)])

    def run_chunk_real(chunk_s):
        n = math.ceil(total_s / chunk_s)
        words_out, lat, dec, lines = [], [], [], []
        for i in range(n):
            a0, a1 = i * chunk_s, min(total_s, (i + 1) * chunk_s)
            ws, ms = transcribe_words(audio[int(a0 * bench.SR):int(a1 * bench.SR)])
            dec.append(ms)
            lines.append(" ".join(text for text, _, _ in ws))
            for text, _, e in ws:
                words_out.append(text)
                lat.append(a1 + ms / 1000 - (a0 + e))
        return words_out, lat, dec, chunk_s, "", lines

    def run_stream_real(stride_s, max_window_s, agreements):
        """Sliding window + LocalAgreement, tracking words by ABSOLUTE TIME (as Whisper-Streaming
        does), not by position in the hypothesis. A re-decode that drops, merges or splits an
        earlier word therefore cannot corrupt or crash the bookkeeping of what is already shown.
        (The first version indexed by position and failed with IndexError when a later decode
        returned fewer words than were already committed.)"""
        TOL = 0.1                          # s: word-start tolerance around the last committed word
        ws_s, committed, lat, dec = 0.0, [], [], []
        state = {"last_end": 0.0}          # absolute end time of the last committed word
        hyps, t, resets, fallbacks = [], 0.0, 0, 0

        def fresh(words):
            """Words of one decode that are not committed yet (absolute times)."""
            new = [w for w in words if w[1] > state["last_end"] - TOL]
            for n in range(min(5, len(committed), len(new)), 0, -1):   # drop a re-emitted boundary word
                if [_key(c[0]) for c in committed[-n:]] == [_key(w[0]) for w in new[:n]]:
                    return new[n:]
            return new

        def commit(words, shown_at):
            for text, s0, e0 in words:
                committed.append((text, s0, e0))
                lat.append(shown_at - e0)
            if words:
                state["last_end"] = words[-1][2]

        while t < total_s:
            t = min(total_s, t + stride_s)
            raw, ms = transcribe_words(audio[int(ws_s * bench.SR):int(t * bench.SR)])
            dec.append(ms)
            cur = [(w, ws_s + s0, ws_s + e0) for w, s0, e0 in raw]
            hyps.append(cur)
            if len(hyps) > agreements:
                hyps.pop(0)
            shown = t + ms / 1000
            if len(hyps) == agreements:
                fr = [fresh(h) for h in hyps]
                n = 0
                while all(n < len(f) for f in fr) and len({_key(f[n][0]) for f in fr}) == 1:
                    n += 1
                commit(fr[-1][:n], shown)
            if t - ws_s > max_window_s:
                resets += 1
                if state["last_end"] > ws_s:
                    ws_s = state["last_end"]        # cut right after the last CONFIRMED word
                else:                               # nothing confirmed in a whole window: flush, hard reset
                    commit(fresh(cur), shown)
                    ws_s = t
                    fallbacks += 1
                hyps.clear()
        raw, ms = transcribe_words(audio[int(ws_s * bench.SR):])
        commit(fresh([(w, ws_s + s0, ws_s + e0) for w, s0, e0 in raw]), total_s + ms / 1000)
        return [c[0] for c in committed], lat, dec, stride_s, f"resets={resets} fallbacks={fallbacks}"

    def report(mode, param, res, agr=""):
        pred, lat, dec, step, note, *extra = res
        save(f"chunk_{param:g}" if mode == "chunk" else f"stream_s{param:g}_a{agr}",
             extra[0] if extra else [" ".join(pred)])
        pn = _norm_tokens(pred)
        s, dl, ins = align_counts(pn, ref_n)
        rtf = (sum(dec) / len(dec)) / (step * 1000) if dec else 0.0
        print(f"{mode:<8}{param:<7g}{agr:<5}{len(pred):<7}{wer(pn, ref_n):<8.3f}{'-' if args.ref_text else format(wer(pred, reference), '.3f'):<8}"
              f"{f'{s}/{dl}/{ins}':<11}{median(lat):<9.2f}{pct(lat, 95):<9.2f}"
              f"{median(dec):<8.0f}{rtf:<6.2f}{note}")

    print(f"{'mode':<8}{'param':<7}{'agr':<5}{'words':<7}{'WER':<8}{'rawWER':<8}{'S/D/I':<11}"
          f"{'lat_med':<9}{'lat_p95':<9}{'dec_ms':<8}{'RTF':<6}notes")
    if args.ref_text:   # the single-pass decode is the ceiling every streaming mode is chasing
        on = _norm_tokens(offline)
        s_, d_, i_ = align_counts(on, ref_n)
        print(f"{'offline':<8}{'-':<7}{'':<5}{len(offline):<7}{wer(on, ref_n):<8.3f}{'-':<8}"
              f"{f'{s_}/{d_}/{i_}':<11}{'-':<9}{'-':<9}{'-':<8}{'-':<6}whole file decoded at once")
    for cs in args.chunk_s:
        report("chunk", cs, run_chunk_real(cs))
    for agr in args.agreements:
        for st in args.stride_s:
            report("stream", st, run_stream_real(st, args.max_window_s, agr), agr)
    print(f"\nreference = {len(ref_n)} normalised words. WER = normalised (case/punctuation ignored); "
          "rawWER = exact tokens, as in the first run.")
    print("S/D/I = substitutions / deletions (reference words missing) / insertions (extra words).")
    print("lat_* = seconds from a word being spoken to it being shown. dec_ms = median decode time.")
    print("RTF = mean decode time / step (chunk or stride); > 1 means the loop cannot keep up in real time.")
    if args.ref_text:
        print("WER is measured against the ground-truth transcript. Numbers, spelled-out numbers and hyphenation can "
              "still differ from Whisper's style; the 'offline' row shows how much error is not the method's fault.")
    else:
        print("WER is measured against Whisper's own offline transcript, not ground truth.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=None,
                    help="run against real audio instead of the synthetic script (needs your GPU)")
    ap.add_argument("--model", default="base.en", help="[--wav mode] faster-whisper model name")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="[--wav mode]")
    ap.add_argument("--save-transcripts", default=None, metavar="DIR",
                    help="[--wav mode] write every mode's transcript to DIR (input for gloss_effect.py)")
    ap.add_argument("--ref-text", default=None,
                    help="[--wav mode] text file with the TRUE transcript; WER is then scored against it")
    ap.add_argument("--src", default="en", help="[--wav mode] speech language")
    ap.add_argument("--decode", default="bounded", choices=["default", "bounded"],
                    help="[--wav mode] see bench.py -- bounded removes latency-spike retries")
    ap.add_argument("--wpm", type=float, default=160.0, help="[sim mode] synthetic speaking rate, words/min")
    ap.add_argument("--chunk-s", type=float, nargs="+", default=[1.0, 2.0],
                    help="chunk lengths to test for the today's-approach baseline")
    ap.add_argument("--stride-s", type=float, nargs="+", default=[0.5, 1.0],
                    help="re-decode intervals to test for sliding-window streaming")
    ap.add_argument("--max-window-s", type=float, default=8.0,
                    help="hard cap on how far the streaming window is allowed to grow")
    ap.add_argument("--agreements", type=int, nargs="+", default=[2],
                    help="LocalAgreement-n: consecutive re-decodes that must agree before a word commits")
    ap.add_argument("--edge-margin", type=float, default=0.25,
                    help="[sim mode] seconds of headroom before a window's end a word needs to be considered settled")
    ap.add_argument("--show", action="store_true", help="[sim mode] print each run's committed transcript")
    args = ap.parse_args()

    if args.wav:
        run_real(args)
        return

    words = build_words(SIM_SCRIPT, args.wpm)
    total = words[-1].end
    print(f"Synthetic script: {len(words)} words, {total:.1f}s at {args.wpm:.0f} wpm "
          f"(true text: {SIM_SCRIPT!r})")

    rows = []
    for cs in args.chunk_s:
        c = run_chunk_baseline(words, cs)
        rows.append(("chunk", cs, summarize(c, words)))
        if args.show:
            print(f"\n[chunk {cs}s] {transcript_of(c)}")
    for st in args.stride_s:
        s = run_streaming(words, st, args.max_window_s, args.agreements[0], args.edge_margin)
        rows.append(("stream", st, summarize(s, words)))
        if args.show:
            print(f"\n[stream stride={st}s, agreements={args.agreements[0]}] {transcript_of(s)}")

    print_table(rows)


if __name__ == "__main__":
    main()