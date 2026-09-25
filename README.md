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

## License

MIT
