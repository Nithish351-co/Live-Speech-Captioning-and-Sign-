"""Rule-based English text -> sign gloss (approximate ISL-style ordering).

APPROXIMATION ONLY: rules reflect commonly described properties (topic/time first,
SOV order, wh-signs last, function words dropped, negation after the verb).
They must be checked against published ISL grammar (e.g. ISLRTC materials) and are
NOT validated by ISL signers.
"""
import json
import os
import re

from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

_lem = WordNetLemmatizer()

BE_FORMS = {"am", "is", "are", "was", "were", "be", "been", "being"}
DROP_TAGS = {"DT", "TO", "IN", "CC", "EX", "POS"}
NEG_WORDS = {"not", "never", "no"}
TIME_WORDS = {"yesterday", "today", "tomorrow", "now", "tonight"}
WH_TAGS = {"WDT", "WP", "WP$", "WRB"}
DO_FORMS = {"do", "does", "did"}
PRONOUNS = {"i": "ME", "me": "ME", "my": "MY", "mine": "MY"}

CONTRACTIONS = [
    (r"\bcan't\b", "can not"), (r"\bwon't\b", "will not"), (r"\bcannot\b", "can not"),
    (r"n't\b", " not"), (r"'m\b", " am"), (r"'re\b", " are"), (r"'ll\b", " will"),
    (r"'ve\b", " have"), (r"'d\b", " would"),
]
_POS_MAP = {"V": "v", "N": "n", "J": "a", "R": "r"}


def _lemma(word: str, tag: str) -> str:
    return _lem.lemmatize(word, _POS_MAP.get(tag[0], "n"))


def _sentence_to_gloss(sentence: str) -> list[str]:
    for pat, rep in CONTRACTIONS:
        sentence = re.sub(pat, rep, sentence, flags=re.IGNORECASE)
    words = re.findall(r"[A-Za-z]+", sentence)
    if not words:
        return []
    tagged = [list(x) for x in pos_tag(words)]

    # Repair: the word after do/does/did (+ not) is a verb; the tagger often mislabels it
    for i, (w, _) in enumerate(tagged):
        if w.lower() in DO_FORMS:
            j = i + 1
            if j < len(tagged) and tagged[j][0].lower() in NEG_WORDS:
                j += 1
            if j < len(tagged) and tagged[j][1] in {"IN", "NN"}:
                tagged[j][1] = "VB"

    items = []  # (lemma_or_gloss, tag)
    for i, (w, t) in enumerate(tagged):
        lw = w.lower()
        if t in DROP_TAGS or lw in BE_FORMS:
            continue
        # drop auxiliary do/does/did when a main verb follows
        if lw in DO_FORMS and any(x[1].startswith("VB") and x[0].lower() not in DO_FORMS
                                  for x in tagged[i + 1:]):
            continue
        items.append((lw, t))

    negated = any(w in NEG_WORDS for w, _ in items)
    items = [(w, t) for w, t in items if w not in NEG_WORDS]

    time_items = [(w, t) for w, t in items if w in TIME_WORDS]
    wh_items = [(w, t) for w, t in items if t in WH_TAGS]
    modal_items = [(w, t) for w, t in items if t == "MD"]
    rest = [(w, t) for w, t in items
            if w not in TIME_WORDS and t not in WH_TAGS and t != "MD"]

    # SOV: move the first main verb to the end of the clause
    for i, (_, t) in enumerate(rest):
        if t.startswith("VB"):
            rest.append(rest.pop(i))
            break

    out = []
    for w, t in time_items + rest:
        out.append(PRONOUNS.get(w) or _lemma(w, t).upper())
    out += [w.upper() for w, _ in modal_items]
    if negated:
        out.append("NOT")
    out += [w.upper() for w, _ in wh_items]
    return out


def text_to_gloss(text: str) -> list[str]:
    gloss: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        gloss += _sentence_to_gloss(sent)
    return gloss


def load_lexicon(path: str | None = None) -> dict:
    path = path or os.path.join(os.path.dirname(__file__), "Lexicon.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)