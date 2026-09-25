from __future__ import annotations

import re

from scoreboard.reading import (
    N_TRAIN_FACT,
    N_TRAIN_Q,
    N_TRAIN_VARIANT,
    NEGATIONS,
    RELATIONS,
    UNKNOWN,
    UPDATES,
)

NAME = r"[A-Z][a-z]+ [A-Z][a-z]+"


def _compile(template: str, subject: str) -> re.Pattern:
    pat = re.escape(template).replace(re.escape("{s}"), f"(?P<s>{subject})")
    # A value may appear twice in one template; the second occurrence must match the first.
    pat = pat.replace(re.escape("{v}"), "(?P<v>.+?)", 1).replace(re.escape("{v}"), "(?P=v)")
    return re.compile(f"^{pat}$")


class RegexReader:
    # Floor (train templates only) or label check (all templates). Knows the generator's wording; understands nothing.
    def __init__(self, all_templates: bool):
        self.name = "floor-regex-all-templates" if all_templates else "floor-regex-train-templates"

        def pick(ts: list[str], n: int) -> list[str]:
            return ts if all_templates else ts[:n]

        self.facts = [
            (r, "fact", _compile(t, NAME)) for r, spec in RELATIONS.items() for t in pick(spec[2], N_TRAIN_FACT)
        ]
        self.facts += [
            (r, "update", _compile(t, NAME)) for r, ts in UPDATES.items() for t in pick(ts, N_TRAIN_VARIANT)
        ]
        self.facts += [(r, "neg", _compile(t, NAME)) for r, ts in NEGATIONS.items() for t in pick(ts, N_TRAIN_VARIANT)]
        self.questions = [(r, _compile(t, ".+?")) for r, spec in RELATIONS.items() for t in pick(spec[3], N_TRAIN_Q)]

    def _records(self, facts: tuple[str, ...]) -> list[tuple[str, str, str, str | None]]:
        out = []
        for f in facts:
            for r, kind, pat in self.facts:
                m = pat.match(f)
                if m:
                    out.append((m.group("s"), r, kind, m.groupdict().get("v")))
                    break
        return out

    def read(self, facts: tuple[str, ...], question: str) -> tuple[str, float]:
        recs = self._records(facts)
        for r, pat in self.questions:
            m = pat.match(question)
            if not m:
                continue
            s = m.group("s")
            hop = re.fullmatch(rf"({NAME})'s (sister|brother)", s)
            if hop:
                links = [
                    v for subj, rel, kind, v in recs if subj == hop.group(1) and rel == hop.group(2) and kind == "fact"
                ]
                if not links:
                    return UNKNOWN, 0.0
                s = links[-1]
            hits = [(kind, v) for subj, rel, kind, v in recs if subj == s and rel == r]
            if not hits or hits[-1][0] == "neg":
                return UNKNOWN, 0.0
            return hits[-1][1], 1.0
        return UNKNOWN, 0.0
