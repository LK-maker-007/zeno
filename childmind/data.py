from __future__ import annotations

import random

from scoreboard.world import N_TRAIN_TEMPLATES, RELATIONS, Fact, make_world

UNKNOWN = "unknown"
MAX_FACTS = 8


def prompt(facts: list[str], question: str) -> str:
    return "".join(f"<f> {t}\n" for t in facts) + f"<q> {question}\n<a> "


def encode(text: str) -> list[int]:
    return list(text.encode())


def decode(ids: list[int]) -> str:
    return bytes(ids).decode(errors="replace")


def _by_subject(facts: list[Fact]) -> dict[str, list[Fact]]:
    out: dict[str, list[Fact]] = {}
    for f in facts:
        out.setdefault(f.subject, []).append(f)
    return out


def example(
    rng: random.Random,
    facts: list[Fact],
    by_subject: dict[str, list[Fact]],
    heldout_question: bool = False,
    p_include: float = 0.7,
) -> tuple[str, str]:
    subject = rng.choice(list(by_subject))
    own = {f.relation: f for f in by_subject[subject]}
    # Half the questions target a relation the subject has, so answerable and "unknown" come out near 50/50.
    relation = rng.choice(list(own)) if rng.random() < 0.5 else rng.choice(list(RELATIONS))
    target = own.get(relation)
    include = target is not None and rng.random() < p_include
    # Distractors: the subject's other facts first, since those make the cat/dog near-miss.
    same = [f.text for r, f in own.items() if r != relation]
    other = [f.text for f in rng.sample(facts, min(len(facts), MAX_FACTS * 2)) if f.subject != subject]
    k = rng.randint(1, MAX_FACTS)
    context = rng.sample(same, min(len(same), rng.randint(0, k))) + other
    context = context[: k - (1 if include else 0)]
    if include:
        context.append(target.text)
    rng.shuffle(context)
    questions = RELATIONS[relation][3]
    q = questions[N_TRAIN_TEMPLATES] if heldout_question else rng.choice(questions[:N_TRAIN_TEMPLATES])
    answer = target.value if include else UNKNOWN
    return prompt(context, q.format(s=subject)), answer


def build(world_seeds: range, n_per_world: int, seed: int, heldout_question: bool = False) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    out = []
    for ws in world_seeds:
        w = make_world(ws)
        by_subject = _by_subject(w.facts)
        out += [example(rng, w.facts, by_subject, heldout_question) for _ in range(n_per_world)]
    return out
