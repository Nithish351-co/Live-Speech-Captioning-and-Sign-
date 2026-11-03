"""Offline simulation of commit policies on a bench_results.csv run.

Live captions show each chunk's text as soon as it is decoded. Translation and sign gloss
work better on whole sentences, so they run on *committed* units. This script replays the
per-chunk transcripts from bench.py under different commit policies and reports the trade-off:

  chunk     every chunk is its own unit (what the pipeline does today)
  window    commit every N chunks
  sentence  commit at a sentence end, when the transcript pauses, or after --max-wait-s

Metrics (no reference transcript needed):
  units / words      how many units, mean words per unit
  complete%          share of units that start with a capital and end at a real sentence end
                     (a rough proxy for "well-formed sentence", NOT translation accuracy)
  forced%            share of units cut by the max-wait limit rather than a sentence end
  wait_mean / wait_max   extra seconds a word waits for its unit to commit, after it was
                     first shown as a live caption (the latency cost of segmenting)

    python segment_eval.py --csv bench_results.csv --label gpu-base.en-bounded --chunk 1 --show sentence
"""
import argparse
import csv
import math
import re

# A sentence cannot really end on these, so a "." after them is a chunk-boundary artifact
DANGLING = {"a", "an", "the", "to", "of", "in", "on", "at", "for", "with", "and", "or", "but",
            "than", "that", "is", "are", "was", "were", "be", "my", "your", "his", "her", "its",
            "their", "our", "this", "these", "those", "if", "because", "so", "by", "from", "as"}


def strip_ellipsis(t: str) -> str:
    return re.sub(r"\s*\.\.\.\s*$", "", t.strip())


def is_terminal(text: str) -> bool:
    t = text.strip()
    if not t or t.endswith("...") or t[-1] not in ".?!":
        return False
    words = re.findall(r"[A-Za-z']+", t)
    return bool(words) and words[-1].lower() not in DANGLING


def is_complete(text: str) -> bool:
    t = text.strip()
    return bool(t) and (t[0].isupper() or t[0].isdigit()) and is_terminal(t)


def load(csv_path, label, chunk_s, mode):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["label"] == label and float(r["chunk_s"]) == chunk_s and r["mode"] == mode:
                rows.append({"i": int(r["chunk"]), "text": r["sample_text"],
                             "shown_ms": (int(r["chunk"]) + 1) * chunk_s * 1000 + float(r["e2e_ms"])})
    return sorted(rows, key=lambda x: x["i"])


def simulate(chunks, policy, max_chunks, window):
    units, buf = [], []

    def commit(at_ms, forced):
        text = " ".join(strip_ellipsis(c["text"]) for c in buf).strip()
        units.append({"text": text, "forced": forced, "commit_ms": at_ms, "n": len(buf),
                      "waits": [at_ms - c["shown_ms"] for c in buf]})
        buf.clear()

    for c in chunks:
        if not c["text"].strip():           # silence: a pause is a natural boundary
            if buf and policy == "sentence":
                commit(buf[-1]["shown_ms"], False)
            continue
        buf.append(c)
        if policy == "chunk":
            commit(c["shown_ms"], False)
        elif policy == "window":
            if len(buf) >= window:
                commit(c["shown_ms"], False)
        else:  # sentence
            if is_terminal(c["text"]):
                commit(c["shown_ms"], False)
            elif len(buf) >= max_chunks:
                commit(c["shown_ms"], True)
    if buf:
        commit(buf[-1]["shown_ms"], True)
    return units


def summarize(units):
    waits = [w for u in units for w in u["waits"]]
    n = max(1, len(units))
    return {"units": len(units),
            "words": sum(len(u["text"].split()) for u in units) / n,
            "complete": 100 * sum(is_complete(u["text"]) for u in units) / n,
            "forced": 100 * sum(u["forced"] for u in units) / n,
            "wait_mean": (sum(waits) / max(1, len(waits))) / 1000,
            "wait_max": (max(waits) if waits else 0) / 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="bench_results.csv")
    ap.add_argument("--label", required=True)
    ap.add_argument("--chunk", type=float, required=True)
    ap.add_argument("--mode", default="parallel")
    ap.add_argument("--max-wait-s", type=float, default=5.0)
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--show", choices=["chunk", "window", "sentence"])
    a = ap.parse_args()

    chunks = load(a.csv, a.label, a.chunk, a.mode)
    if not chunks:
        raise SystemExit("no rows match --label/--chunk/--mode")
    max_chunks = max(1, math.ceil(a.max_wait_s / a.chunk))
    window = max(1, math.ceil(a.window_s / a.chunk))

    print(f"{a.label}, {a.chunk:g} s chunks, {len(chunks)} chunks | window={window} chunks, "
          f"max wait={max_chunks} chunks")
    print(f"\n{'policy':<10}{'units':<7}{'words':<8}{'complete%':<11}{'forced%':<9}{'wait_mean':<11}wait_max")
    results = {}
    for p in ("chunk", "window", "sentence"):
        units = simulate(chunks, p, max_chunks, window)
        results[p] = units
        s = summarize(units)
        print(f"{p:<10}{s['units']:<7}{s['words']:<8.1f}{s['complete']:<11.0f}{s['forced']:<9.0f}"
              f"{s['wait_mean']:<11.2f}{s['wait_max']:.1f} s")
    if a.show:
        print(f"\nCommitted units ({a.show}):")
        for u in results[a.show]:
            print(f"  {'[forced] ' if u['forced'] else ''}{u['text']}")


if __name__ == "__main__":
    main()