from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from childmind.model import ChildMind  # noqa: E402
from childmind.model import Config as C1Config  # noqa: E402
from mind.data import collate  # noqa: E402
from mind.model import Config, FastWeightReader, decode, delta_write, loss_fn  # noqa: E402
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


def test_a_lone_example_with_no_facts_is_read_not_crashed():
    torch.manual_seed(0)
    model = FastWeightReader(Config(hidden=32)).eval()
    b = collate([Example((), "Who is Ana Reddy's brother?", UNKNOWN, "simple", "Ana Reddy")], "cpu")
    assert b["facts"].shape[1] == 1
    assert not b["fact_mask"].any()
    with torch.no_grad():
        out = model(b)
    assert torch.isfinite(out["unknown_logit"]).all()
    [(text, conf)] = decode(out, b["facts"])
    assert isinstance(text, str)
    assert 0.0 <= conf <= 1.0


def test_race_adapter_loads_a_training_checkpoint_and_reads(tmp_path):
    from dataclasses import asdict

    from scoreboard.read_race import MindReader

    torch.manual_seed(0)
    c = Config(hidden=32)
    ckpt = tmp_path / "a1-s0" / "mind" / "last.pt"
    ckpt.parent.mkdir(parents=True)
    # Same layout as mind.train writes.
    torch.save({"model": FastWeightReader(c).state_dict(), "config": asdict(c), "step": 0, "metrics": {}}, ckpt)
    reader = MindReader(ckpt, threads=1)
    assert reader.name == "A1-fast-weight (a1-s0)"
    for facts in ((), ("Ana Reddy lives in Pune.", "Ana Reddy has a cat called Milo.")):
        text, conf = reader.read(facts, "Where does Ana Reddy live?")
        assert isinstance(text, str)
        assert 0.0 <= conf <= 1.0


def test_a_later_write_with_the_same_key_overwrites_the_earlier_value():
    k = torch.nn.functional.normalize(torch.tensor([[1.0, 2.0, 0.0, 0.0]]), dim=-1)
    other = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    v1, v2 = torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.tensor([[0.0, 5.0, 0.0, 0.0]])
    full = torch.ones(1)
    m = delta_write(delta_write(torch.zeros(1, 4, 4), k, v1, full), k, v2, full)
    assert torch.allclose(torch.einsum("bij,bj->bi", m, k), v2, atol=1e-6)
    assert torch.allclose(torch.einsum("bij,bj->bi", m, other), torch.zeros(1, 4), atol=1e-6)
    p = delta_write(delta_write(torch.zeros(1, 4, 4), k, k, full), k, k, full)
    assert torch.allclose(torch.einsum("bij,bj->bi", p, k), k, atol=1e-6)
    assert torch.allclose(torch.einsum("bij,bj->bi", p, other), torch.zeros(1, 4), atol=1e-6)
