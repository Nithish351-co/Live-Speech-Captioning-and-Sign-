"""Gloss quality evaluation harness.

Runs gloss.py's text_to_gloss() against gloss_test_set.json and reports, per
sentence and aggregated:
  - exact_match        whole gloss sequence identical to the reference
  - wer                token-level edit distance / len(reference)  (order errors)
  - f1 / precision/recall   bag-of-words overlap, ignoring order (word-choice errors)
  - lex_hits / n        how many predicted gloss tokens have a clip in lexicon.json

Splitting WER (order-sensitive) from F1 (order-insensitive) matters: a low F1
means the rules picked the wrong words; a high F1 with a high WER means the
words are right but the reordering logic put them in the wrong place. Those
need different fixes.

Usage:
    python setup_nltk.py        # once, if not already done
    python eval_gloss.py        # dev set: prints a report, writes gloss_eval_results.csv
    python eval_gloss.py --set gloss_heldout_set.json --out gloss_heldout_results.csv   # held-out set

Run the held-out set ONCE and report what comes out; do not tune gloss.py against it.

This was written and traced against gloss.py's rules but NOT run end-to-end in
this environment (no network access to fetch NLTK's tagger data here). Run it
yourself and treat any case that disagrees with its reference_gloss as useful
signal, not as this script being broken -- see each case's "confidence" and
"note" fields in gloss_test_set.json before assuming the rule is at fault.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

from gloss import load_lexicon, text_to_gloss

TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "gloss_test_set.json")
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "gloss_eval_results.csv")


def levenshtein(a: list[str], b: list[str]) -> int:
    """Token-level edit distance (insertions/deletions/substitutions all cost 1)."""
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


def bag_prf1(pred: list[str], ref: list[str]) -> tuple[float, float, float]:
    """Precision/recall/F1 over multisets of tokens, ignoring order.

    Edge-case convention: precision is 1.0 when pred is empty and ref is also
    empty (nothing predicted, nothing to get wrong), else 0.0 if pred is empty
    but ref isn't. Symmetrically, recall is 1.0 whenever ref is empty (there
    was nothing to miss), regardless of what was predicted.
    """
    common = sum((Counter(pred) & Counter(ref)).values())
    precision = common / len(pred) if pred else (1.0 if not ref else 0.0)
    recall = common / len(ref) if ref else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate(cases: list[dict], lexicon: dict) -> list[dict]:
    rows = []
    for case in cases:
        pred = text_to_gloss(case["text"])
        ref = case["reference_gloss"]
        wer = levenshtein(pred, ref) / max(1, len(ref))
        precision, recall, f1 = bag_prf1(pred, ref)
        hits = sum(1 for g in pred if g in lexicon)
        rows.append({
            "id": case["id"],
            "text": case["text"],
            "confidence": case.get("confidence", "?"),
            "rules_tested": "|".join(case.get("rules_tested", [])),
            "reference": " ".join(ref),
            "predicted": " ".join(pred),
            "exact_match": pred == ref,
            "wer": round(wer, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "lex_hits": hits,
            "gloss_n": len(pred),
            "note": case.get("note", ""),
        })
    return rows


def print_report(rows: list[dict]) -> None:
    print(f"{'id':<22} {'exact':<6} {'wer':<6} {'f1':<6} {'lex':<6} conf")
    print("-" * 70)
    for r in rows:
        print(f"{r['id']:<22} {str(r['exact_match']):<6} {r['wer']:<6} {r['f1']:<6} "
              f"{r['lex_hits']}/{r['gloss_n']:<4} {r['confidence']}")
        if not r["exact_match"]:
            print(f"    text:      {r['text']}")
            print(f"    reference: {r['reference']}")
            print(f"    predicted: {r['predicted']}")
            if r["note"]:
                print(f"    note:      {r['note']}")

    n = len(rows)
    exact_rate = sum(r["exact_match"] for r in rows) / n
    mean_wer = sum(r["wer"] for r in rows) / n
    mean_f1 = sum(r["f1"] for r in rows) / n
    total_hits = sum(r["lex_hits"] for r in rows)
    total_gloss = sum(r["gloss_n"] for r in rows)

    print("\n" + "=" * 70)
    print(f"Overall (n={n}): exact-match={exact_rate:.1%}  mean WER={mean_wer:.3f}  "
          f"mean F1={mean_f1:.3f}  lexicon coverage={total_hits}/{total_gloss} "
          f"({total_hits / max(1, total_gloss):.1%})")

    # Breakdown by rule tag -- shows which ISL grammar rule is most error-prone,
    # the number worth putting in the paper's "gloss quality" section.
    by_rule: dict[str, list[dict]] = {}
    for r in rows:
        for tag in r["rules_tested"].split("|"):
            by_rule.setdefault(tag, []).append(r)
    print("\nBy rule tag:")
    print(f"{'rule':<18} {'n':<4} {'exact%':<8} {'mean_wer':<10} mean_f1")
    for tag, rs in sorted(by_rule.items()):
        n2 = len(rs)
        print(f"{tag:<18} {n2:<4} {sum(r['exact_match'] for r in rs) / n2:<8.1%} "
              f"{sum(r['wer'] for r in rs) / n2:<10.3f} {sum(r['f1'] for r in rs) / n2:.3f}")

    low_conf = [r for r in rows if r["confidence"] == "low"]
    if low_conf:
        print(f"\n{len(low_conf)} case(s) marked low-confidence in the test set "
              "(tagger-dependent edge cases) -- check these by hand before trusting the score:")
        for r in low_conf:
            print(f"  - {r['id']}: {r['text']}")


def write_csv(rows: list[dict], path: str = RESULTS_CSV) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {path}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate gloss.py against a JSON test set.")
    ap.add_argument("--set", default=TEST_SET_PATH, help="test-set JSON (default: gloss_test_set.json)")
    ap.add_argument("--out", default=RESULTS_CSV, help="results CSV (default: gloss_eval_results.csv)")
    args = ap.parse_args()
    print(f"Test set: {args.set}")
    with open(args.set, encoding="utf-8") as f:
        data = json.load(f)
    lexicon = load_lexicon()
    try:
        rows = evaluate(data["cases"], lexicon)
    except LookupError as e:
        sys.exit(f"NLTK data missing: run `python setup_nltk.py` first.\n{e}")
    print_report(rows)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()