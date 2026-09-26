from __future__ import annotations

import math
import random

import pytest

torch = pytest.importorskip("torch")

from ears.data import batches, index, pad  # noqa: E402
from ears.features import frames, log_mel, mask, mel_filterbank  # noqa: E402
from ears.model import Config, QuartzNet  # noqa: E402
from ears.score import edits, error_rate, speaker_bootstrap  # noqa: E402
from ears.text import INDEX, SYMBOLS, encode, greedy  # noqa: E402


def test_size_matches_quartznet_5x5():
    # Kriman et al. 2020, Table 1: QuartzNet-5x5 has 6.7M parameters.
    assert abs(QuartzNet(Config()).n_params() / 6.7e6 - 1) < 0.01


def test_output_is_half_the_frames_and_finite():
    torch.manual_seed(0)
    model = QuartzNet(Config()).eval()
    x = torch.randn(2, 64, 101)
    logits, out = model(x, torch.tensor([101, 60]))
    assert logits.shape == (2, 29, 51)
    assert out.tolist() == [51, 30]
    assert torch.isfinite(logits).all()


def test_every_mel_filter_has_weight():
    fb = mel_filterbank()
    assert fb.shape == (64, 257)
    assert (fb.sum(1) > 0).all()
    assert fb.max() <= 1.0


def test_log_mel_frame_count_and_normalisation():
    wave = torch.sin(torch.arange(16000 * 2) * 0.05) + 0.01 * torch.randn(16000 * 2)
    x = log_mel(wave, mel_filterbank())
    assert x.shape == (64, frames(16000 * 2))
    assert torch.allclose(x.mean(1), torch.zeros(64), atol=1e-4)


def test_greedy_collapses_repeats_but_keeps_letters_split_by_blank():
    path = ["H", "H", "<blank>", "E", "L", "L", "<blank>", "L", "O", "O", "<blank>"]
    logits = torch.full((len(SYMBOLS), len(path)), -10.0)
    for t, s in enumerate(path):
        logits[INDEX[s], t] = 10.0
    assert greedy(logits, len(path)) == "HELLO"
    assert greedy(logits, 3) == "H"
    assert encode("IT'S A") == [INDEX[c] for c in "IT'S A"]


def test_word_and_char_error_rates():
    assert edits(list("kitten"), list("sitting")) == 3
    assert error_rate(["THE CAT SAT"], ["THE CAT SAT"], "word") == (0, 3)
    assert error_rate(["THE CAT SAT"], ["A CAT"], "word") == (2, 3)
    assert error_rate(["AB"], ["AC"], "char") == (1, 2)


def test_speaker_bootstrap_centres_on_the_pooled_rate():
    errors, lengths = [1, 0, 3, 2, 0, 1], [10, 10, 10, 10, 10, 10]
    speakers = ["a", "a", "b", "b", "c", "c"]
    rate, lo, hi = speaker_bootstrap(errors, lengths, speakers)
    assert rate == pytest.approx(7 / 60)
    assert lo <= rate <= hi
    assert (rate, lo, hi) == speaker_bootstrap(errors, lengths, speakers)


def test_time_masks_stay_inside_each_utterance():
    x = torch.ones(2, 64, 200)
    x[1, :, 100:] = 7.0
    mask(x, torch.tensor([200, 100]), torch.Generator().manual_seed(0), time=50)
    assert (x[1, :, 100:] == 7.0).all()
    assert (x == 0).any()


def test_batches_cover_every_utterance_once():
    lengths = [random.Random(i).randint(50, 500) for i in range(103)]
    chunks = batches(lengths, 10, random.Random(0))
    assert sorted(i for c in chunks for i in c) == list(range(103))
    assert len(chunks) == math.ceil(103 / 10)


def test_pad_and_ctc_learn_a_tiny_batch():
    torch.manual_seed(0)
    model = QuartzNet(Config())
    items = [
        (torch.randn(64, 120).half(), torch.tensor(encode("HI THERE"))),
        (torch.randn(64, 90).half(), torch.tensor(encode("YES"))),
    ]
    x, xl, y, yl = pad(items, "cpu")
    assert x.shape == (2, 64, 120) and xl.tolist() == [120, 90] and yl.tolist() == [8, 3]
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    losses = []
    for _ in range(25):
        logits, ol = model(x, xl)
        loss = torch.nn.functional.ctc_loss(logits.log_softmax(1).permute(2, 0, 1), y, ol, yl, zero_infinity=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_index_reads_librispeech_layout(tmp_path):
    d = tmp_path / "LibriSpeech" / "dev-clean" / "84" / "121123"
    d.mkdir(parents=True)
    (d / "84-121123.trans.txt").write_text(
        "84-121123-0000 GO DO YOU HEAR\n84-121123-0001 BUT IN LESS THAN FIVE MINUTES\n"
    )
    other = tmp_path / "LibriSpeech" / "test-clean" / "1" / "2"
    other.mkdir(parents=True)
    (other / "1-2.trans.txt").write_text("1-2-0000 NOT THIS ONE\n")
    utts = index(tmp_path, ["dev-clean"])
    assert [u.text for u in utts] == ["GO DO YOU HEAR", "BUT IN LESS THAN FIVE MINUTES"]
    assert utts[0].audio == d / "84-121123-0000.flac"
    assert {u.speaker for u in utts} == {"84"}
