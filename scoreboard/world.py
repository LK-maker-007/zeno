from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass

FIRST = [
    "Priya",
    "Arun",
    "Meena",
    "Karthik",
    "Divya",
    "Rahul",
    "Anita",
    "Vikram",
    "Lakshmi",
    "Suresh",
    "Nila",
    "Ravi",
    "Kavya",
    "Hari",
    "Deepa",
    "Manoj",
    "Sita",
    "Ganesh",
    "Uma",
    "Rohan",
    "Asha",
    "Kumar",
    "Farah",
    "Omar",
    "Lena",
    "Tomas",
    "Mei",
    "Jun",
    "Sara",
    "Diego",
    "Ines",
    "Yusuf",
    "Ana",
    "Ivan",
    "Noor",
    "Emil",
    "Zara",
    "Leo",
    "Hana",
    "Kofi",
]
LAST = [
    "Raman",
    "Iyer",
    "Nair",
    "Das",
    "Khan",
    "Silva",
    "Chen",
    "Park",
    "Ali",
    "Costa",
    "Menon",
    "Rao",
    "Pillai",
    "Gupta",
    "Sato",
    "Kim",
    "Novak",
    "Haddad",
    "Mensah",
    "Lopez",
    "Bose",
    "Joshi",
    "Reddy",
    "Singh",
    "Tanaka",
    "Ortiz",
    "Petrov",
    "Yilmaz",
    "Okafor",
    "Moreau",
]
PET_NAMES = [
    "Mochi",
    "Bruno",
    "Luna",
    "Tiger",
    "Coco",
    "Max",
    "Bella",
    "Rocky",
    "Pepper",
    "Simba",
    "Daisy",
    "Oreo",
    "Milo",
    "Kiwi",
    "Ginger",
    "Shadow",
    "Biscuit",
    "Nemo",
    "Pixel",
    "Toffee",
]
CITIES = [
    "Chennai",
    "Madurai",
    "Mumbai",
    "Pune",
    "Delhi",
    "Kochi",
    "Lisbon",
    "Seoul",
    "Nairobi",
    "Lima",
    "Oslo",
    "Cairo",
    "Osaka",
    "Porto",
    "Accra",
    "Quito",
    "Dublin",
    "Hanoi",
    "Tunis",
    "Perth",
]
JOBS = [
    "teacher",
    "nurse",
    "farmer",
    "pilot",
    "baker",
    "doctor",
    "engineer",
    "painter",
    "driver",
    "chef",
    "lawyer",
    "tailor",
    "plumber",
    "writer",
    "dentist",
    "singer",
    "carpenter",
    "librarian",
    "potter",
    "sailor",
]
FOODS = ["dosa", "biryani", "pasta", "sushi", "tacos", "noodles", "idli", "pizza", "falafel", "dumplings"]
COLORS = ["blue", "green", "red", "yellow", "purple", "orange", "black", "white", "pink", "brown"]
SPORTS = ["cricket", "football", "tennis", "chess", "hockey", "badminton", "swimming", "cycling", "running", "kabaddi"]

# Relations in one family share wording, so a missing relation next to a present one is a near-miss.
RELATIONS = {
    "cat_name": (
        "pets",
        PET_NAMES,
        [
            "{s}'s cat is named {v}.",
            "{s} has a cat called {v}.",
            "The cat that lives with {s} is {v}.",
            "{s} named their cat {v}.",
        ],
        [
            "What is {s}'s cat called?",
            "What is the name of {s}'s cat?",
            "What did {s} name their cat?",
            "Which name does the cat of {s} have?",
        ],
    ),
    "dog_name": (
        "pets",
        PET_NAMES,
        [
            "{s}'s dog is named {v}.",
            "{s} has a dog called {v}.",
            "The dog that lives with {s} is {v}.",
            "{s} named their dog {v}.",
        ],
        [
            "What is {s}'s dog called?",
            "What is the name of {s}'s dog?",
            "What did {s} name their dog?",
            "Which name does the dog of {s} have?",
        ],
    ),
    "rabbit_name": (
        "pets",
        PET_NAMES,
        [
            "{s}'s rabbit is named {v}.",
            "{s} has a rabbit called {v}.",
            "The rabbit that lives with {s} is {v}.",
            "{s} named their rabbit {v}.",
        ],
        [
            "What is {s}'s rabbit called?",
            "What is the name of {s}'s rabbit?",
            "What did {s} name their rabbit?",
            "Which name does the rabbit of {s} have?",
        ],
    ),
    "sister_name": (
        "family",
        FIRST,
        [
            "{s}'s sister is {v}.",
            "{s} has a sister named {v}.",
            "{v} is the sister of {s}.",
            "The sister of {s} is called {v}.",
        ],
        [
            "Who is {s}'s sister?",
            "What is the name of {s}'s sister?",
            "What is {s}'s sister called?",
            "Which person is the sister of {s}?",
        ],
    ),
    "brother_name": (
        "family",
        FIRST,
        [
            "{s}'s brother is {v}.",
            "{s} has a brother named {v}.",
            "{v} is the brother of {s}.",
            "The brother of {s} is called {v}.",
        ],
        [
            "Who is {s}'s brother?",
            "What is the name of {s}'s brother?",
            "What is {s}'s brother called?",
            "Which person is the brother of {s}?",
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
        ],
        [
            "Where does {s} live?",
            "Which city does {s} live in?",
            "In what city is {s}'s home?",
            "What city is home to {s}?",
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
        ],
        [
            "Where does {s} work?",
            "Which city does {s} work in?",
            "In what city is {s}'s office?",
            "What city does {s} commute to for work?",
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
        ],
        [
            "Where was {s} born?",
            "Which city is {s}'s birthplace?",
            "In what city was {s} born?",
            "What city did {s} come into the world in?",
        ],
    ),
    "job": (
        "work",
        JOBS,
        ["{s} works as a {v}.", "{s} is a {v}.", "{s} earns a living as a {v}.", "By profession, {s} is a {v}."],
        [
            "What does {s} do for work?",
            "What is {s}'s job?",
            "How does {s} earn a living?",
            "What is the profession of {s}?",
        ],
    ),
    "favorite_food": (
        "likes",
        FOODS,
        [
            "{s}'s favourite food is {v}.",
            "{s} loves eating {v}.",
            "{s} would always choose {v}.",
            "The food {s} likes best is {v}.",
        ],
        [
            "What is {s}'s favourite food?",
            "What food does {s} love?",
            "Which dish would {s} always choose?",
            "What food does {s} like best?",
        ],
    ),
    "favorite_color": (
        "likes",
        COLORS,
        [
            "{s}'s favourite colour is {v}.",
            "{s} loves the colour {v}.",
            "{s} would always pick {v}.",
            "The colour {s} likes best is {v}.",
        ],
        [
            "What is {s}'s favourite colour?",
            "Which colour does {s} love?",
            "What colour would {s} always pick?",
            "What colour does {s} like best?",
        ],
    ),
    "favorite_sport": (
        "likes",
        SPORTS,
        [
            "{s}'s favourite sport is {v}.",
            "{s} loves playing {v}.",
            "{s} spends weekends on {v}.",
            "The sport {s} likes best is {v}.",
        ],
        [
            "What is {s}'s favourite sport?",
            "Which sport does {s} love playing?",
            "What does {s} spend weekends on?",
            "What sport does {s} like best?",
        ],
    ),
}
N_TRAIN_TEMPLATES = 3


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    value: str
    text: str
    source: str
    time: int
    batch: int


@dataclass(frozen=True)
class Probe:
    subject: str
    relation: str
    question: str
    gold: str | None
    stratum: str


@dataclass
class World:
    seed: int
    facts: list[Fact]
    probes: list[Probe]
    heldout_statements: list[str]

    def to_jsonl(self) -> str:
        lines = [json.dumps({"kind": "meta", "seed": self.seed})]
        lines += [json.dumps({"kind": "fact", **asdict(f)}) for f in self.facts]
        lines += [json.dumps({"kind": "probe", **asdict(p)}) for p in self.probes]
        lines += [json.dumps({"kind": "heldout", "text": t}) for t in self.heldout_statements]
        return "\n".join(lines) + "\n"

    def digest(self) -> str:
        return hashlib.sha256(self.to_jsonl().encode()).hexdigest()[:16]


def make_world(
    seed: int = 0, n_people: int = 150, p_relation: float = 0.45, n_batches: int = 4, probes_per_stratum: int = 300
) -> World:
    rng = random.Random(seed)
    names = [f"{first} {last}" for first in FIRST for last in LAST]
    rng.shuffle(names)
    people, unseen = names[:n_people], names[n_people : n_people * 2]

    held: dict[str, dict[str, str]] = {}
    for s in people:
        held[s] = {r: rng.choice(spec[1]) for r, spec in RELATIONS.items() if rng.random() < p_relation}
        if not held[s]:
            r = rng.choice(list(RELATIONS))
            held[s][r] = rng.choice(RELATIONS[r][1])

    pairs = [(s, r, v) for s in people for r, v in held[s].items()]
    rng.shuffle(pairs)
    facts, heldout = [], []
    for i, (s, r, v) in enumerate(pairs):
        statements = RELATIONS[r][2]
        text = rng.choice(statements[:N_TRAIN_TEMPLATES]).format(s=s, v=v)
        facts.append(
            Fact(
                s,
                r,
                v,
                text,
                source=f"user, session {1 + i * n_batches // len(pairs)}",
                time=i,
                batch=i * n_batches // len(pairs),
            )
        )
        heldout.append(statements[N_TRAIN_TEMPLATES].format(s=s, v=v))

    def question(r: str, s: str, heldout_template: bool) -> str:
        qs = RELATIONS[r][3]
        return (qs[N_TRAIN_TEMPLATES] if heldout_template else rng.choice(qs[:N_TRAIN_TEMPLATES])).format(s=s)

    def family_of(r: str) -> str:
        return RELATIONS[r][0]

    pools: dict[str, list[Probe]] = {
        k: [] for k in ["i_known", "ii_near_miss", "ii_far_miss", "iii_unseen_subject", "iv_paraphrase"]
    }
    for s, r, v in pairs:
        pools["i_known"].append(Probe(s, r, question(r, s, False), v, "i_known"))
        pools["iv_paraphrase"].append(Probe(s, r, question(r, s, True), v, "iv_paraphrase"))
    for s in people:
        present_families = {family_of(r) for r in held[s]}
        for r in RELATIONS:
            if r in held[s]:
                continue
            stratum = "ii_near_miss" if family_of(r) in present_families else "ii_far_miss"
            pools[stratum].append(Probe(s, r, question(r, s, False), None, stratum))
    for s in unseen:
        r = rng.choice(list(RELATIONS))
        pools["iii_unseen_subject"].append(Probe(s, r, question(r, s, False), None, "iii_unseen_subject"))

    probes = []
    for k in pools:
        rng.shuffle(pools[k])
        probes += pools[k][:probes_per_stratum]
    return World(seed, facts, probes, heldout)


def normalize(text: str | None) -> str:
    if text is None:
        return ""
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch == " ").strip()
