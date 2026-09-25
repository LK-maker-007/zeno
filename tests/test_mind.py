from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from childmind.model import ChildMind  # noqa: E402
from childmind.model import Config as C1Config  # noqa: E402
from mind.data import collate  # noqa: E402
from mind.model import Config, FastWeightReader, decode, loss_fn  # noqa: E402
from scoreboard.reading import CATEGORIES, UNKNOWN, Example, build  # noqa: E402


def test_size_within_ten_percent_of_the_transformer_baseline():
    ratio = FastWeightReader(Config()).n_params() / ChildMind(C1Config()).n_params()
    assert 0.9 <= ratio <= 1.1


def test_spans_point_at_the_answer_bytes():
    data = [e for e in build(range(1000, 1002), 5, CATEGORIES, seed=0) if e.answer != UNKNOWN]
    b = collate(data, "cpu")
    assert b["uncopyable"] == 0
    for i, e in enumerate(data):
        j, s = (int(x) for x in b["start"][i].nonzero()[0])
        t = int(b["end"][i, j].nonzero()[0])
        assert bytes(b["facts"][i, j, s : t + 1].tolist()).decode() == e.answer


def test_forward_and_backward_stay_finite_with_an_empty_context():
    torch.manual_seed(0)
    model = FastWeightReader(Config(hidden=32))
    data = build(range(1000, 1001), 2, CATEGORIES, seed=0)
    data.append(Example((), "Who is Ana Reddy's brother?", UNKNOWN, "simple", "Ana Reddy"))
    b = collate(data, "cpu")
    out = model(b)
    loss = loss_fn(out, b)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for text, conf in decode(out, b["facts"]):
        assert isinstance(text, str)
        assert 0.0 <= conf <= 1.0
