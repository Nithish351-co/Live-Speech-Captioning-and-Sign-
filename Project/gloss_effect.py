"""How much does the ASR chunking strategy damage the SIGN output?

For each recording, every mode's transcript (from `stream_bench.py --save-transcripts DIR`)
is turned into gloss and compared with the gloss of the single-pass OFFLINE transcript of the
same audio. So this isolates the effect of chunking/streaming on the gloss; ASR mistakes that
every mode shares (the offline decode's own errors) cancel out.

    python stream_bench.py --wav speech.wav --save-transcripts transcripts --model base.en --device cuda --chunk-s 1 2 --stride-s 1 2 --agreements 1 2 3
    python stream_bench.py --wav clips\\ls_121_121726.wav --ref-text clips\\ls_121_121726.txt --save-transcripts transcripts ...
    python gloss_effect.py --dir transcripts

How each transcript is glossed (mirrors the live server):
  chunk_*    one line per chunk, and each line is glossed on its own (the server glosses each chunk
             separately, so a sentence cut by a chunk boundary is glossed as two fragments)
  stream_*   one continuous text of committed words, glossed sentence by sentence

Columns: asr_wer = word error of that mode's text vs the offline text (case/punctuation ignored);
gloss_wer / gloss_f1 = token-level edit distance / bag-of-words F1 of the gloss vs the offline gloss.
gloss_wer is order-sensitive, so it also catches wrongly reordered clauses; f1 only catches wrong words.

Also prints lexicon coverage of the offline gloss: the share of gloss tokens that have a sign clip,
on real speech (not on test sentences written to hit the lexicon), plus the most common missing glosses.
"""
import argparse
import csv
import glob
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from eval_gloss import bag_prf1, levenshtein
    from gloss import load_lexicon, text_to_gloss
    from stream_bench import _norm_tokens
except LookupError as e:
    sys.exit(f"NLTK data missing: run `python setup_nltk.py` first.\n{e}")

MODE_ORDER = re.compile(r"^(chunk|stream)_(?:s)?([\d.]+)(?:_a(\d+))?$")


def mode_sort_key(name: str):
    m = MODE_ORDER.match(name)
    if not m:
        return (9, 0, 0)
    return (0 if m.group(1) == "chunk" else 1, int(m.group(3) or 0), float(m.group(2)))


def gloss_of(lines):
    out = []
    for line in lines:
        out += text_to_gloss(line)
    return out


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="transcripts")
    ap.add_argument("--out", default="gloss_effect_results.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*__*.txt")))
    stems = sorted({os.path.basename(f).split("__")[0] for f in files})
    if not stems:
        sys.exit(f"No transcripts found in {args.dir}. Run stream_bench.py with --save-transcripts {args.dir} first.")

    lexicon = {k.upper() for k in load_lexicon()}
    rows, ref_gloss_all = [], []
    for stem in stems:
        off_path = os.path.join(args.dir, f"{stem}__offline.txt")
        if not os.path.exists(off_path):
            print(f"{stem}: no offline transcript, skipped")
            continue
        off_lines = read_lines(off_path)
        ref_gloss = gloss_of(off_lines)
        ref_gloss_all += ref_gloss
        ref_words = _norm_tokens(" ".join(off_lines).split())
        modes = sorted((os.path.basename(f)[len(stem) + 2:-4] for f in files
                        if os.path.basename(f).startswith(stem + "__")), key=mode_sort_key)
        for mode in modes:
            if mode in ("offline", "truth"):
                continue
            lines = read_lines(os.path.join(args.dir, f"{stem}__{mode}.txt"))
            gloss = gloss_of(lines)
            p, r, f1 = bag_prf1(gloss, ref_gloss)
            pred_words = _norm_tokens(" ".join(lines).split())
            rows.append({"recording": stem, "mode": mode,
                         "asr_wer": round(levenshtein(pred_words, ref_words) / max(1, len(ref_words)), 3),
                         "gloss_wer": round(levenshtein(gloss, ref_gloss) / max(1, len(ref_gloss)), 3),
                         "gloss_f1": round(f1, 3), "gloss_tokens": len(gloss),
                         "ref_gloss_tokens": len(ref_gloss),
                         "units": sum(1 for ln in lines if ln.strip())})

    for stem in stems:
        rs = [r for r in rows if r["recording"] == stem]
        if not rs:
            continue
        print(f"\n{stem}  (reference gloss = offline transcript, {rs[0]['ref_gloss_tokens']} tokens)")
        print(f"{'mode':<16}{'asr_wer':<9}{'gloss_wer':<11}{'gloss_f1':<10}{'gloss_tok':<10}units")
        for r in rs:
            print(f"{r['mode']:<16}{r['asr_wer']:<9.3f}{r['gloss_wer']:<11.3f}{r['gloss_f1']:<10.3f}"
                  f"{r['gloss_tokens']:<10}{r['units']}")

    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
    print(f"\nMean over {len(stems)} recording(s)  [min-max]")
    print(f"{'mode':<16}{'asr_wer':<9}{'gloss_wer':<24}{'gloss_f1'}")
    for mode in sorted(by_mode, key=mode_sort_key):
        rs = by_mode[mode]
        gw = [r["gloss_wer"] for r in rs]
        print(f"{mode:<16}{sum(r['asr_wer'] for r in rs) / len(rs):<9.3f}"
              f"{sum(gw) / len(gw):.3f} [{min(gw):.3f}-{max(gw):.3f}]      "
              f"{sum(r['gloss_f1'] for r in rs) / len(rs):.3f}")

    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.out}")

    if ref_gloss_all:
        hits = sum(1 for g in ref_gloss_all if g in lexicon)
        print(f"\nLexicon coverage on real speech (offline gloss): {hits}/{len(ref_gloss_all)} tokens "
              f"({hits / len(ref_gloss_all):.1%}) have a sign clip; lexicon has {len(lexicon)} entries.")
        miss = Counter(g for g in ref_gloss_all if g not in lexicon)
        print("Most frequent glosses WITHOUT a clip (record these first): "
              + ", ".join(f"{w}({n})" for w, n in miss.most_common(15)))
        print("Note: gloss tokens include function-like words such as pronouns, so read coverage together with "
              "the by-token list above.")


if __name__ == "__main__":
    main()