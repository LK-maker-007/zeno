# Zeno

[![CI](https://github.com/LK-maker-007/zeno/actions/workflows/ci.yml/badge.svg)](https://github.com/LK-maker-007/zeno/actions/workflows/ci.yml)

The codebase of **Ava Lovelace Zeno**: a mind built from its own parts, meant to run on a laptop CPU, keep
learning without forgetting, and say "I don't know" instead of guessing.

Nothing here is borrowed as a trained model. Existing models appear only as opponents on a scoreboard, and
every part of Ava has to beat its opponent on that scoreboard before it replaces anything.

## Scoreboard

`scoreboard/` generates a small synthetic world of people, pets, family, places and jobs, tells its facts to
an entrant, and scores what the entrant says back:

| Test | Measures |
|---|---|
| T1 | Recall of facts it was told, including question wordings it never saw |
| T2 | Forgetting of early facts as later facts arrive |
| T3 | Whether its confidence separates right answers from near-misses and unknown people (AUROC, selective accuracy, AURC) |
| T5 | Cost per fact and per question |

Every run starts with a self-check: an oracle control must score AUROC 1.000 and a random-confidence control
must land near 0.5, or the run stops.

```bash
pip install -e .[dev]
python -m scoreboard.run --seed 0
```

## Reading benchmark

`scoreboard/reading.py` tests a mind on reading, not recall: a handful of facts and a question, answered from
the facts or with "unknown". Nine categories:

| Category | What it checks |
|---|---|
| `simple` | Plain lookup, or "unknown" when the fact is absent |
| `near_miss` | The person has a fact of the same kind ("cat"), the question asks for another ("dog") |
| `same_name` | Another person sharing a first or last name has the asked-for fact |
| `update` | A later fact ("now lives in Pune") replaces an earlier one |
| `negation` | "has no dog" means the answer is "unknown" |
| `two_hop` | "Where does Priya's sister live?" needs two facts |
| `heldout_question` | Question wordings never used in training |
| `heldout_fact` | Fact wordings never used in training |
| `heldout_both` | Both at once |

Each relation has 6 fact wordings and 8 question wordings. Two fact wordings and three question wordings per
relation are held out, so the held-out categories test meaning, not memorised phrasing.

Two regex readers set the floor. One knows only the training wordings: it scores 0.000 on held-out answerable
questions, which is what a template-matcher earns there. The other knows every wording, and the tests require
it to agree with every gold label, which is how the labels are checked.

Accuracy on "unknown" questions alone rewards a reader that cannot parse anything; read it together with
answerable accuracy.

```bash
python -m scoreboard.read_race --n 100
```

## C1: the baseline mind

`childmind/` is the first mind and the opponent every later one must beat at equal size: a 4.9M-parameter
byte-level Transformer (d 256, 6 layers, 8 heads, 512-byte context), written and trained from scratch. It is a
known architecture on purpose. A new architecture can only be shown better against a fair, working baseline.

It learns to read, not to memorise: every training example is a few facts from a world it will never be
tested on, a question, and the answer or "unknown", with the same person's other facts as distractors.

```bash
pip install -e .[dev,model]
python -m childmind.train --task reading   # scoreboard v2 data; --task v1 reproduces run 1
```

Run 1 (v1 data, Kaggle T4, seed 0, 6,000 steps) failed its gate of 0.95: answer accuracy 0.906 and "unknown"
accuracy 0.915 on unseen worlds, 0.276 on held-out question wording. Most errors answered from another
person's fact of the same kind. That result is also why the reading benchmark exists: the v1 task was
solvable by template matching, so the gate measured parsing, not reading.

`kaggle/c1/` holds the Kaggle kernel that ran it.

## License

MIT
