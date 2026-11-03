"""Run on your laptop (next to gloss.py): shows what each rehearsal sentence turns into on the phone.
    python check_demo.py                 # built-in sentences
    python check_demo.py "Hello how are you" "Please help me"
Tokens marked [sign] play an emoji sign; [spell] tokens are fingerspelled. Re-word sentences until
most tokens are [sign]; do NOT change gloss.py for this (it is already evaluated in the paper)."""
import re, sys
from gloss import text_to_gloss

html = open("static/phone.html", encoding="utf-8").read()
block = html[html.index("const EMOJI"):html.index("const ALIAS")]
SIGNS = set(re.findall(r"\b([A-Z]+):\s*\"", block))
SENTENCES = sys.argv[1:] or [
    "Hello, how are you?", "Thank you.", "Please help me.", "I want water.",
    "Sorry, I do not understand.", "What is your name?", "Good morning, my friend.",
    "Yes, I need the doctor.", "I am not happy today.", "Where is the hospital?",
]
tot = hit = 0
for s in SENTENCES:
    g = text_to_gloss(s)
    marks = " ".join(f"{t}[{'sign' if t in SIGNS else 'spell'}]" for t in g)
    n = sum(t in SIGNS for t in g); tot += len(g); hit += n
    print(f"{s!r:38} -> {marks}")
print(f"\n{hit}/{tot} gloss tokens have a sign ({hit / max(1, tot):.0%})")