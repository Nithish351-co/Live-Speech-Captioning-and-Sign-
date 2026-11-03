"""Run once: python setup_nltk.py"""
import nltk

for pkg in ["averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "wordnet", "omw-1.4"]:
    ok = nltk.download(pkg, quiet=True)
    print(f"{pkg}: {'ok' if ok else 'not available (fine if the other tagger name worked)'}")