from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass

from scoreboard.world import CITIES, COLORS, FIRST, FOODS, JOBS, LAST, PET_NAMES, SPORTS

UNKNOWN = "unknown"
N_TRAIN_FACT, N_TRAIN_Q = 4, 5

# Each relation: family (for near-misses), value pool (None = another person), fact templates, question templates.
# The first N_TRAIN_FACT / N_TRAIN_Q templates are for training; the rest are held out and never trained on.
RELATIONS: dict[str, tuple[str, list[str] | None, list[str], list[str]]] = {
    "cat": (
        "pets",
        PET_NAMES,
        [
            "{s}'s cat is named {v}.",
            "{s} has a cat called {v}.",
            "The cat that lives with {s} is {v}.",
            "{s} named their cat {v}.",
            "{v} is the name of {s}'s cat.",
            "{s} owns a cat, and its name is {v}.",
        ],
        [
            "What is {s}'s cat called?",
            "What is the name of {s}'s cat?",
            "What did {s} name their cat?",
            "Which name does the cat of {s} have?",
            "What do you call {s}'s cat?",
            "Tell me the name of {s}'s cat.",
            "{s}'s cat goes by what name?",
            "How is {s}'s cat named?",
        ],
    ),
    "dog": (
        "pets",
        PET_NAMES,
        [
            "{s}'s dog is named {v}.",
            "{s} has a dog called {v}.",
            "The dog that lives with {s} is {v}.",
            "{s} named their dog {v}.",
            "{v} is the name of {s}'s dog.",
            "{s} owns a dog, and its name is {v}.",
        ],
        [
            "What is {s}'s dog called?",
            "What is the name of {s}'s dog?",
            "What did {s} name their dog?",
            "Which name does the dog of {s} have?",
            "What do you call {s}'s dog?",
            "Tell me the name of {s}'s dog.",
            "{s}'s dog goes by what name?",
            "How is {s}'s dog named?",
        ],
    ),
    "rabbit": (
        "pets",
        PET_NAMES,
        [
            "{s}'s rabbit is named {v}.",
            "{s} has a rabbit called {v}.",
            "The rabbit that lives with {s} is {v}.",
            "{s} named their rabbit {v}.",
            "{v} is the name of {s}'s rabbit.",
            "{s} owns a rabbit, and its name is {v}.",
        ],
        [
            "What is {s}'s rabbit called?",
            "What is the name of {s}'s rabbit?",
            "What did {s} name their rabbit?",
            "Which name does the rabbit of {s} have?",
            "What do you call {s}'s rabbit?",
            "Tell me the name of {s}'s rabbit.",
            "{s}'s rabbit goes by what name?",
            "How is {s}'s rabbit named?",
        ],
    ),
    "sister": (
        "family",
        None,
        [
            "{s}'s sister is {v}.",
            "{s} has a sister named {v}.",
            "{v} is the sister of {s}.",
            "The sister of {s} is called {v}.",
            "{s} and {v} are sisters, {v} being the sister.",
            "{s} grew up with a sister, {v}.",
        ],
        [
            "Who is {s}'s sister?",
            "What is the name of {s}'s sister?",
            "What is {s}'s sister called?",
            "Which person is the sister of {s}?",
            "Who is the sister of {s}?",
            "Tell me who {s}'s sister is.",
            "{s}'s sister is who?",
            "Name the sister of {s}.",
        ],
    ),
    "brother": (
        "family",
        None,
        [
            "{s}'s brother is {v}.",
            "{s} has a brother named {v}.",
            "{v} is the brother of {s}.",
            "The brother of {s} is called {v}.",
            "{s} and {v} are siblings, {v} being the brother.",
            "{s} grew up with a brother, {v}.",
        ],
        [
            "Who is {s}'s brother?",
            "What is the name of {s}'s brother?",
            "What is {s}'s brother called?",
            "Which person is the brother of {s}?",
            "Who is the brother of {s}?",
            "Tell me who {s}'s brother is.",
            "{s}'s brother is who?",
            "Name the brother of {s}.",
        ],
    ),
    "lives_in": (
        "places",
        CITIES,
        [
            "{s} lives in {v}.",
            "{s}'s home is in {v}.",
            "{s} has made a home in {v}.",
            "The city where {s} lives is {v}.",
            "{v} is where {s} lives.",
            "{s} resides in {v}.",
        ],
        [
            "Where does {s} live?",
            "Which city does {s} live in?",
            "In what city is {s}'s home?",
            "What city is home to {s}?",
            "Where is {s}'s home?",
            "Tell me where {s} lives.",
            "{s} lives in which city?",
            "Which city is {s} living in?",
        ],
    ),
    "works_in": (
        "places",
        CITIES,
        [
            "{s} works in {v}.",
            "{s}'s office is in {v}.",
            "{s} commutes to work in {v}.",
            "The city where {s} works is {v}.",
            "{v} is where {s} works.",
            "{s} has a workplace in {v}.",
        ],
        [
            "Where does {s} work?",
            "Which city does {s} work in?",
            "In what city is {s}'s office?",
            "What city does {s} commute to for work?",
            "Where is {s}'s office?",
            "Tell me where {s} works.",
            "{s} works in which city?",
            "Which city is {s} working in?",
        ],
    ),
    "born_in": (
        "places",
        CITIES,
        [
            "{s} was born in {v}.",
            "{s}'s birthplace is {v}.",
            "{s} came into the world in {v}.",
            "The city where {s} was born is {v}.",
            "{v} is where {s} was born.",
            "{s} originally comes from {v}.",
        ],
        [
            "Where was {s} born?",
            "Which city is {s}'s birthplace?",
            "In what city was {s} born?",
            "What city did {s} come into the world in?",
            "What is {s}'s birthplace?",
            "Tell me where {s} was born.",
            "{s} was born in which city?",
            "Which city is {s} from originally?",
        ],
    ),
    "job": (
        "work",
        JOBS,
        [
            "{s} works as a {v}.",
            "{s} is a {v}.",
            "{s} earns a living as a {v}.",
            "By profession, {s} is a {v}.",
            "{s} has a job as a {v}.",
            "{s} makes money working as a {v}.",
        ],
        [
            "What does {s} do for work?",
            "What is {s}'s job?",
            "How does {s} earn a living?",
            "What is the profession of {s}?",
            "What is {s}'s profession?",
            "Tell me what {s} does for a living.",
            "{s} works as what?",
            "What kind of work does {s} do?",
        ],
    ),
    "food": (
        "likes",
        FOODS,
        [
            "{s}'s favourite food is {v}.",
            "{s} loves eating {v}.",
            "{s} would always choose {v}.",
            "The food {s} likes best is {v}.",
            "{v} is {s}'s favourite food.",
            "{s} enjoys {v} more than any food.",
        ],
        [
            "What is {s}'s favourite food?",
            "What food does {s} love?",
            "Which dish would {s} always choose?",
            "What food does {s} like best?",
            "What does {s} most like to eat?",
            "Tell me {s}'s favourite food.",
            "{s}'s favourite food is what?",
            "Which food is {s}'s favourite?",
        ],
    ),
    "color": (
        "likes",
        COLORS,
        [
            "{s}'s favourite colour is {v}.",
            "{s} loves the colour {v}.",
            "{s} would always pick {v}.",
            "The colour {s} likes best is {v}.",
            "{v} is {s}'s favourite colour.",
            "{s} likes {v} more than any colour.",
        ],
        [
            "What is {s}'s favourite colour?",
            "Which colour does {s} love?",
            "What colour would {s} always pick?",
            "What colour does {s} like best?",
            "What is the colour {s} prefers?",
            "Tell me {s}'s favourite colour.",
            "{s}'s favourite colour is what?",
            "Which colour is {s}'s favourite?",
        ],
    ),
    "sport": (
        "likes",
        SPORTS,
        [
            "{s}'s favourite sport is {v}.",
            "{s} loves playing {v}.",
            "{s} spends weekends on {v}.",
            "The sport {s} likes best is {v}.",
            "{v} is {s}'s favourite sport.",
            "{s} enjoys {v} more than any sport.",
        ],
        [
            "What is {s}'s favourite sport?",
            "Which sport does {s} love playing?",
            "What does {s} spend weekends on?",
            "What sport does {s} like best?",
            "What sport does {s} prefer?",
            "Tell me {s}'s favourite sport.",
            "{s}'s favourite sport is what?",
            "Which sport is {s}'s favourite?",
        ],
    ),
}
UPDATES = {
    "lives_in": ["{s} has since moved to {v}.", "{s} now lives in {v}.", "{s} recently relocated to {v}."],
    "works_in": ["{s} has since changed offices to {v}.", "{s} now works in {v}.", "{s} recently moved jobs to {v}."],
    "job": ["{s} has since become a {v}.", "{s} now works as a {v}.", "{s} recently retrained as a {v}."],
}
NEGATIONS = {
    "cat": ["{s} has no cat.", "{s} does not own a cat.", "{s} has never had a cat."],
    "dog": ["{s} has no dog.", "{s} does not own a dog.", "{s} has never had a dog."],
    "rabbit": ["{s} has no rabbit.", "{s} does not own a rabbit.", "{s} has never had a rabbit."],
    "sister": ["{s} has no sister.", "{s} does not have a sister.", "{s} never had a sister."],
    "brother": ["{s} has no brother.", "{s} does not have a brother.", "{s} never had a brother."],
}
N_TRAIN_VARIANT = 2
HOP_TARGETS = ["lives_in", "works_in", "born_in", "job"]
CATEGORIES = [
    "simple",
    "near_miss",
    "same_name",
    "update",
    "negation",
    "two_hop",
    "heldout_question",
    "heldout_fact",
    "heldout_both",
]
MAX_FACTS = 8
# Training never sees held-out wording; those categories exist only for validation and test.
TRAIN_CATEGORIES = [c for c in CATEGORIES if not c.startswith("heldout")]


@dataclass(frozen=True)
class Example:
    facts: tuple[str, ...]
    question: str
    answer: str
    category: str
    subject: str


def _fact(rng: random.Random, s: str, r: str, v: str, heldout: bool) -> str:
    t = RELATIONS[r][2]
    return rng.choice(t[N_TRAIN_FACT:] if heldout else t[:N_TRAIN_FACT]).format(s=s, v=v)


def _question(rng: random.Random, s: str, r: str, heldout: bool) -> str:
    t = RELATIONS[r][3]
    return rng.choice(t[N_TRAIN_Q:] if heldout else t[:N_TRAIN_Q]).format(s=s)


def _variant(rng: random.Random, table: dict[str, list[str]], r: str, heldout: bool, **kw) -> str:
    t = table[r]
    return rng.choice(t[N_TRAIN_VARIANT:] if heldout else t[:N_TRAIN_VARIANT]).format(**kw)


class Reading:
    def __init__(self, seed: int, n_people: int = 120, p_relation: float = 0.5):
        self.seed = seed
        rng = random.Random(seed)
        names = [f"{a} {b}" for a in FIRST for b in LAST]
        rng.shuffle(names)
        self.people = names[:n_people]
        self.facts: dict[str, dict[str, str]] = {}
        for s in self.people:
            held = {}
            for r, (_, pool, _, _) in RELATIONS.items():
                if rng.random() < p_relation:
                    held[r] = rng.choice(pool) if pool else rng.choice([p for p in self.people if p != s])
            self.facts[s] = held

    def _others(self, rng: random.Random, avoid: set[str], k: int, heldout: bool) -> list[str]:
        out = []
        for _ in range(k * 3):
            o = rng.choice(self.people)
            if o not in avoid and self.facts[o]:
                r = rng.choice(list(self.facts[o]))
                out.append(_fact(rng, o, r, self.facts[o][r], heldout))
            if len(out) >= k:
                break
        return out

    def _pad(
        self,
        rng: random.Random,
        s: str,
        core: list[str],
        heldout: bool,
        exclude: set[str],
        avoid: frozenset = frozenset(),
    ) -> tuple[str, ...]:
        # Distractors: the subject's other facts first (near-misses), then other people's facts.
        own = [_fact(rng, s, r, v, heldout) for r, v in self.facts[s].items() if r not in exclude]
        room = rng.randint(len(core), MAX_FACTS) - len(core)
        extra = rng.sample(own, min(len(own), rng.randint(0, room))) if own else []
        extra += self._others(rng, {s} | avoid, room - len(extra), heldout)
        ctx = core + extra[: max(0, room)]
        rng.shuffle(ctx)
        return tuple(ctx)

    def example(self, rng: random.Random, category: str) -> Example:
        s = rng.choice(self.people)
        own = self.facts[s]
        hq = category in ("heldout_question", "heldout_both")
        hf = category in ("heldout_fact", "heldout_both")
        if category in ("simple", "heldout_question", "heldout_fact", "heldout_both"):
            if own and rng.random() < 0.5:
                r = rng.choice(list(own))
                facts = self._pad(rng, s, [_fact(rng, s, r, own[r], hf)], hf, {r})
                return Example(facts, _question(rng, s, r, hq), own[r], category, s)
            r = rng.choice(list(RELATIONS))
            return Example(self._pad(rng, s, [], hf, {r}), _question(rng, s, r, hq), UNKNOWN, category, s)
        if category == "near_miss":
            fams: dict[str, list[str]] = {}
            for rr in own:
                fams.setdefault(RELATIONS[rr][0], []).append(rr)
            options = [
                (rr, m)
                for fam, present in fams.items()
                for rr in present
                for m in RELATIONS
                if RELATIONS[m][0] == fam and m not in own
            ]
            if not options:
                return self.example(rng, "simple")
            rr, missing = rng.choice(options)
            facts = self._pad(rng, s, [_fact(rng, s, rr, own[rr], False)], False, {rr, missing})
            return Example(facts, _question(rng, s, missing, False), UNKNOWN, category, s)
        if category == "same_name":
            first, last = s.split()
            twins = [p for p in self.people if p != s and (p.split()[0] == first or p.split()[1] == last)]
            shared = [(t, r) for t in twins for r in self.facts[t]]
            if not shared:
                return self.example(rng, "simple")
            t, r = rng.choice(shared)
            core = [_fact(rng, t, r, self.facts[t][r], False)]
            answerable = r in own and rng.random() < 0.5
            if answerable:
                core.append(_fact(rng, s, r, own[r], False))
            facts = self._pad(rng, s, core, False, {r})
            return Example(facts, _question(rng, s, r, False), own[r] if answerable else UNKNOWN, category, s)
        if category == "update":
            r = rng.choice(list(UPDATES))
            pool = RELATIONS[r][1]
            old = own.get(r) or rng.choice(pool)
            new = rng.choice([v for v in pool if v != old])
            first = _fact(rng, s, r, old, False)
            later = _variant(rng, UPDATES, r, False, s=s, v=new)
            rest = self._pad(rng, s, [], False, {r})
            # Order matters: the update must come after the original fact.
            facts = (*rest[: len(rest) // 2], first, *rest[len(rest) // 2 :], later)[-MAX_FACTS:]
            if first not in facts:
                facts = (first, *facts[-(MAX_FACTS - 1) :])
            return Example(tuple(facts), _question(rng, s, r, False), new, category, s)
        if category == "negation":
            r = rng.choice(list(NEGATIONS))
            neg = _variant(rng, NEGATIONS, r, False, s=s)
            facts = self._pad(rng, s, [neg], False, {r})
            return Example(facts, _question(rng, s, r, False), UNKNOWN, category, s)
        if category == "two_hop":
            links = [(fr, own[fr]) for fr in ("sister", "brother") if fr in own]
            if not links:
                return self.example(rng, "simple")
            fr, sib = rng.choice(links)
            target = rng.choice(HOP_TARGETS)
            core = [_fact(rng, s, fr, sib, False)]
            answer = UNKNOWN
            if target in self.facts[sib] and rng.random() < 0.6:
                core.append(_fact(rng, sib, target, self.facts[sib][target], False))
                answer = self.facts[sib][target]
            q = _question(rng, f"{s}'s {fr}", target, False)
            # The sibling is kept out of the random distractors, or its target fact could leak into an "unknown" case.
            return Example(self._pad(rng, s, core, False, {fr}, frozenset({sib})), q, answer, category, s)
        raise ValueError(category)


def build(world_seeds: range, n_per_category: int, categories: list[str], seed: int) -> list[Example]:
    rng = random.Random(seed)
    out = []
    for ws in world_seeds:
        w = Reading(ws)
        out += [w.example(rng, c) for c in categories for _ in range(n_per_category)]
    rng.shuffle(out)
    return out


def digest(examples: list[Example]) -> str:
    return hashlib.sha256("\n".join(json.dumps(asdict(e)) for e in examples).encode()).hexdigest()[:16]
