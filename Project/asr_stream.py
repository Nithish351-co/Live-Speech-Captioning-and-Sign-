"""Context-aware streaming transcript commits for overlapping Whisper decodes."""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np

Word = tuple[str, float, float]
Decoder = Callable[[np.ndarray, str], tuple[Sequence[Word], str]]


def _word_key(word: str) -> str:
    return re.sub(r"[^\w']+", "", word.casefold())


class StreamingASR:
    def __init__(
        self,
        decode: Decoder,
        sample_rate: int = 16000,
        agreements: int = 2,
        max_window_seconds: float = 8.0,
    ) -> None:
        self._decode = decode
        self._sample_rate = sample_rate
        self._agreements = agreements
        self._max_window_seconds = max_window_seconds
        self._audio = np.empty(0, dtype=np.float32)
        self._window_start = 0.0
        self._total_seconds = 0.0
        self._last_end = 0.0
        self._hypotheses: list[list[Word]] = []
        self._committed: list[Word] = []
        self._detected_language = ""

    def process(self, audio: np.ndarray, language: str) -> tuple[list[str], str]:
        self._audio = np.concatenate((self._audio, audio.astype(np.float32, copy=False)))
        self._total_seconds += len(audio) / self._sample_rate

        decode_language = language
        if decode_language == "auto" and self._detected_language:
            decode_language = self._detected_language
        words, detected_language = self._decode(self._audio, decode_language)
        if detected_language:
            self._detected_language = detected_language

        current = [
            (word.strip(), self._window_start + start, self._window_start + end)
            for word, start, end in words
            if word.strip()
        ]
        self._hypotheses.append(current)
        if len(self._hypotheses) > self._agreements:
            self._hypotheses.pop(0)

        fresh_hypotheses = [self._fresh(hypothesis) for hypothesis in self._hypotheses]
        fresh = fresh_hypotheses[-1]
        agreed_count = 0
        if len(fresh_hypotheses) == self._agreements:
            while all(agreed_count < len(hypothesis) for hypothesis in fresh_hypotheses):
                keys = {
                    _word_key(hypothesis[agreed_count][0])
                    for hypothesis in fresh_hypotheses
                }
                if len(keys) != 1:
                    break
                agreed_count += 1

        committed_now = fresh[:agreed_count]
        if committed_now:
            self._commit(committed_now)
            self._trim_to(self._last_end)
        elif self._total_seconds - self._window_start > self._max_window_seconds:
            if fresh:
                self._commit(fresh)
                self._trim_to(self._last_end)
            else:
                self._trim_to(self._total_seconds)
            self._hypotheses.clear()

        return [word[0] for word in committed_now], self._detected_language

    def flush(self, language: str) -> tuple[list[str], str]:
        if not len(self._audio):
            return [], self._detected_language
        decode_language = language
        if decode_language == "auto" and self._detected_language:
            decode_language = self._detected_language
        words, detected_language = self._decode(self._audio, decode_language)
        if detected_language:
            self._detected_language = detected_language
        current = [
            (word.strip(), self._window_start + start, self._window_start + end)
            for word, start, end in words
            if word.strip()
        ]
        pending = self._fresh(current)
        if pending:
            self._commit(pending)
            self._trim_to(self._last_end)
        self._hypotheses.clear()
        return [word[0] for word in pending], self._detected_language

    def _fresh(self, words: Sequence[Word]) -> list[Word]:
        fresh = [word for word in words if word[1] > self._last_end - 0.1]
        for count in range(min(5, len(self._committed), len(fresh)), 0, -1):
            committed_keys = [_word_key(word[0]) for word in self._committed[-count:]]
            fresh_keys = [_word_key(word[0]) for word in fresh[:count]]
            if committed_keys == fresh_keys:
                return fresh[count:]
        return fresh

    def _commit(self, words: Sequence[Word]) -> None:
        self._committed.extend(words)
        self._committed = self._committed[-5:]
        self._last_end = words[-1][2]

    def _trim_to(self, timestamp: float) -> None:
        discard = min(
            len(self._audio),
            max(0, int((timestamp - self._window_start) * self._sample_rate)),
        )
        self._audio = self._audio[discard:]
        self._window_start += discard / self._sample_rate
