"""Build benchmark clips (16 kHz mono WAV + true transcript) from a LibriSpeech folder.

1. Download test-clean.tar.gz (346 MB) from https://www.openslr.org/12 and extract it.
   Check the licence (LibriSpeech is CC BY 4.0) and cite Panayotov et al. (2015).
2. pip install soundfile      (or have ffmpeg on your PATH; either works for reading FLAC)
3. python make_librispeech_clips.py --root path\\to\\LibriSpeech\\test-clean --out clips --n 2

Each clip joins consecutive utterances from ONE speaker/chapter (no utterance is cut in half)
until it reaches about --seconds, with a short silence between utterances. Different speakers
are used for different clips. --seed makes the choice reproducible: report it in the paper.

Outputs, per clip:  ls_<speaker>_<chapter>.wav   and   ls_<speaker>_<chapter>.txt (true transcript)
Then:  python stream_bench.py --wav clips\\ls_1089_134686.wav --ref-text clips\\ls_1089_134686.txt ...
"""
import argparse
import os
import random
import shutil
import subprocess
import sys
import wave

import numpy as np

SR = 16000
GAP_S = 0.25   # silence inserted between joined utterances


def read_audio(path: str) -> np.ndarray:
    """-> mono float32 at 16 kHz"""
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
    except ImportError:
        if not shutil.which("ffmpeg"):
            sys.exit("Need either `pip install soundfile` or ffmpeg on PATH to read FLAC files.")
        proc = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                              capture_output=True, check=True)
        return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != SR:
        n_new = int(len(data) * SR / sr)
        data = np.interp(np.linspace(0, 1, n_new, endpoint=False),
                         np.linspace(0, 1, len(data), endpoint=False), data).astype(np.float32)
    return data


def write_wav(path: str, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def find_chapters(root: str) -> dict:
    """{speaker: [(chapter, chapter_dir), ...]} for every folder holding a *.trans.txt"""
    out: dict = {}
    for spk in sorted(os.listdir(root)):
        sdir = os.path.join(root, spk)
        if not os.path.isdir(sdir):
            continue
        for chap in sorted(os.listdir(sdir)):
            cdir = os.path.join(sdir, chap)
            if os.path.isfile(os.path.join(cdir, f"{spk}-{chap}.trans.txt")):
                out.setdefault(spk, []).append((chap, cdir))
    return out


def build_clip(spk: str, chap: str, cdir: str, seconds: float):
    """Join consecutive utterances from the start of the chapter until reaching `seconds`,
    without exceeding it by more than 20% (an utterance is never cut)."""
    with open(os.path.join(cdir, f"{spk}-{chap}.trans.txt"), encoding="utf-8") as f:
        utts = [line.strip().split(" ", 1) for line in f if line.strip()]
    parts, words, total = [], [], 0.0
    gap = np.zeros(int(GAP_S * SR), dtype=np.float32)
    for uid, text in sorted(utts):
        a = read_audio(os.path.join(cdir, f"{uid}.flac"))
        dur = len(a) / SR
        if parts and total + GAP_S + dur > seconds * 1.2:
            break
        if parts:
            parts.append(gap)
            total += GAP_S
        parts.append(a)
        total += dur
        words.append(text)
        if total >= seconds:
            break
    return np.concatenate(parts), " ".join(words), total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the extracted LibriSpeech/test-clean folder")
    ap.add_argument("--out", default="clips")
    ap.add_argument("--n", type=int, default=2, help="number of clips (different speakers)")
    ap.add_argument("--seconds", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"--root folder not found: {args.root}")
    chapters = find_chapters(args.root)
    if not chapters:
        sys.exit(f"No LibriSpeech speaker/chapter folders found under {args.root}. "
                 "Point --root at the folder that contains the numbered speaker directories.")
    rng = random.Random(args.seed)
    speakers = sorted(chapters)
    rng.shuffle(speakers)
    os.makedirs(args.out, exist_ok=True)

    made = 0
    for spk in speakers:
        if made >= args.n:
            break
        chap, cdir = rng.choice(chapters[spk])
        audio, text, dur = build_clip(spk, chap, cdir, args.seconds)
        if dur < args.seconds * 0.6:
            print(f"speaker {spk}: only {dur:.0f}s available, skipping")
            continue
        base = os.path.join(args.out, f"ls_{spk}_{chap}")
        write_wav(base + ".wav", audio)
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"{base}.wav  {dur:.1f}s  {len(text.split())} words")
        made += 1
    if made < args.n:
        print(f"warning: only produced {made} of {args.n} clips")
    print(f"\nseed={args.seed}. Keep the seed and file names; report them in the paper.")


if __name__ == "__main__":
    main()