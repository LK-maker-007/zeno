from __future__ import annotations

import json

import pytest

from arena.games import HELD_OUT, load_games, split
from arena.score import budget, game_score, level_score

PUBLIC_TITLES = [
    "AR25", "BP35", "CD82", "CN04", "DC22", "FT09", "G50T", "KA59", "LF52", "LP85", "LS20", "M0R0", "R11L",
    "RE86", "S5I5", "SB26", "SC25", "SK48", "SP80", "SU15", "TN36", "TR87", "TU93", "VC33", "WA30",
]  # fmt: skip


def test_level_score_follows_the_published_examples():
    assert level_score(10, 10) == 1.0
    assert level_score(10, 20) == 0.25
    assert level_score(10, 100) == pytest.approx(0.01)
    assert level_score(10, 5) == 1.15


def test_unfinished_game_is_capped_by_the_levels_it_completed():
    assert game_score((10, 10, 10, 10, 10), [5, 5, 5, 5]) == pytest.approx(1.15 * 10 / 15)
    assert game_score((10, 10, 10, 10, 10), [10, 10, 10, 10]) == pytest.approx(10 / 15)
    assert game_score((10, 10, 10, 10, 10), []) == 0.0
    assert game_score((10, 10), [1, 1]) == 1.0


def test_budget_is_five_times_the_human_baseline():
    assert budget(22) == 110


def test_held_out_set_is_every_other_sorted_title(tmp_path):
    for i, title in enumerate(PUBLIC_TITLES):
        d = tmp_path / title.lower() / f"v{i}"
        d.mkdir(parents=True)
        (d / "metadata.json").write_text(
            json.dumps({"game_id": f"{title.lower()}-v{i}", "title": title, "baseline_actions": [1, 2]})
        )
    games = load_games(tmp_path)
    assert frozenset(sorted(PUBLIC_TITLES)[1::2]) == HELD_OUT
    assert len(split(games, "heldout")) == 12
    assert len(split(games, "practice")) == 13
    assert {g.title for g in split(games, "heldout")}.isdisjoint(g.title for g in split(games, "practice"))
