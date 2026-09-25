from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Protocol

from scoreboard.world import Fact, Probe


class Entrant(Protocol):
    name: str

    def tell(self, text: str, source: str, time: int) -> None: ...

    def ask(self, question: str) -> tuple[str, float]: ...


class Oracle:
    # Control: sees the structured world, so it must score perfectly. Proves the scorer can reach 1.0.
    name = "control-oracle"

    def __init__(self, facts: list[Fact], probes: list[Probe]):
        self._by_q = {p.question: (p.subject, p.relation) for p in probes}
        self._facts = {(f.subject, f.relation): f.value for f in facts}
        self._text_to_key = {f.text: (f.subject, f.relation) for f in facts}
        self._told: set[tuple[str, str]] = set()

    def tell(self, text: str, source: str, time: int) -> None:
        self._told.add(self._text_to_key[text])

    def ask(self, question: str) -> tuple[str, float]:
        key = self._by_q[question]
        if key in self._told:
            return self._facts[key], 1.0
        return "unknown", 0.0


class CoinFlip(Oracle):
    # Control: the oracle's answers with random confidence. Both classes occur, so AUROC must sit near 0.5.
    name = "control-coinflip"

    def __init__(self, facts: list[Fact], probes: list[Probe], seed: int = 0):
        super().__init__(facts, probes)
        self._rng = random.Random(seed)

    def ask(self, question: str) -> tuple[str, float]:
        return super().ask(question)[0], self._rng.random()


_WORD = re.compile(r"[A-Za-z']+")
_STOP = {
    "the",
    "a",
    "an",
    "is",
    "of",
    "in",
    "what",
    "who",
    "which",
    "does",
    "do",
    "did",
    "where",
    "how",
    "has",
    "have",
    "their",
    "named",
    "called",
    "name",
    "was",
    "as",
    "by",
    "to",
    "for",
    "on",
    "that",
    "with",
    "s",
    "lives",
    "live",
    "city",
    "person",
    "would",
    "always",
    "best",
    "likes",
    "like",
    "loves",
    "love",
}


def _char_ngrams(text: str, n: int = 3) -> Counter:
    t = f"  {text.lower()}  "
    return Counter(t[i : i + n] for i in range(len(t) - n + 1))


def _cosine(a: Counter, b: Counter) -> float:
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class LexicalCosine:
    # Opponent: one fused similarity score over stored sentences (LMLM-style single threshold signal).
    name = "opponent-lexical-cosine"

    def __init__(self):
        self._store: list[tuple[str, Counter]] = []

    def tell(self, text: str, source: str, time: int) -> None:
        self._store.append((text, _char_ngrams(text)))

    def ask(self, question: str) -> tuple[str, float]:
        if not self._store:
            return "unknown", 0.0
        q = _char_ngrams(question)
        best_text, best = max(((t, _cosine(q, g)) for t, g in self._store), key=lambda x: x[1])
        q_words = {w.lower() for w in _WORD.findall(question)}
        candidates = [
            w
            for w in _WORD.findall(best_text)
            if w.lower() not in q_words and w.lower() not in _STOP and not w.lower().endswith("'s")
        ]
        return (candidates[-1] if candidates else "unknown"), best
