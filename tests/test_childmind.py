from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from childmind.data import UNKNOWN, build, prompt  # noqa: E402
from childmind.infer import answer  # noqa: E402
from childmind.model import ChildMind, Config  # noqa: E402


def test_default_size_is_the_recorded_baseline():
    assert ChildMind(Config()).n_params() == 4_935_680


def test_examples_are_balanced_and_answers_come_from_the_facts():
    data = build(range(1000, 1003), 200, seed=0)
    unknown = sum(a == UNKNOWN for _, a in data) / len(data)
    assert 0.4 < unknown < 0.6
    for p, a in data:
        if a != UNKNOWN:
            assert a in p.split("<q>")[0]


def test_answer_returns_text_and_a_probability():
    torch.manual_seed(0)
    model = ChildMind(Config(d=32, layers=1, heads=2)).eval()
    text, conf = answer(model, prompt(["Ana Reddy lives in Pune."], "Where does Ana Reddy live?"))
    assert isinstance(text, str)
    assert 0.0 <= conf <= 1.0
