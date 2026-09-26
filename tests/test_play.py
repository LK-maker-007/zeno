from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from play.see import objects, place


def test_objects_are_same_colour_4_connected_shapes_on_the_background():
    f = np.zeros((6, 6), np.int8)
    f[0, 0] = f[0, 1] = 3
    f[1, 1] = 3
    f[4, 4] = 3
    f[2:4, 3] = 5
    got = sorted((o.color, o.size) for o in objects(f))
    assert got == [(3, 1), (3, 3), (5, 2)]


def test_a_click_target_lies_on_a_hollow_object():
    f = np.zeros((7, 7), np.int8)
    f[1:6, 1] = f[1:6, 5] = f[1, 1:6] = f[5, 1:6] = 4
    [ring] = objects(f)
    y, x = ring.target()
    assert f[y, x] == 4


def test_place_ignores_the_counter_rows_only():
    a = np.zeros((64, 64), np.int8)
    b, c = a.copy(), a.copy()
    b[0, 5] = b[63, 63] = 7
    c[10, 10] = 7
    assert place(a) == place(b)
    assert place(a) != place(c)


pytest.importorskip("arcengine")

from arcengine.enums import GameAction  # noqa: E402

from play.ava import Ava  # noqa: E402


@dataclass
class Obs:
    frame: np.ndarray
    levels_completed: int
    available: tuple[int, ...]


class Line:
    # Toy world: a marker on a line of 4 cells. ACTION1 steps right (stuck at the end), ACTION2 does nothing,
    # ACTION3 from the last cell loses the level and puts the marker back at the start.
    def __init__(self):
        self.pos = 0

    def obs(self) -> Obs:
        f = np.zeros((64, 64), np.int8)
        f[10, 10 + self.pos] = 3
        return Obs(f, 0, (1, 2, 3))

    def step(self, action: GameAction) -> None:
        if action == GameAction.RESET or (action == GameAction.ACTION3 and self.pos == 3):
            self.pos = 0
        elif action == GameAction.ACTION1:
            self.pos = min(3, self.pos + 1)


def play(steps: int) -> list[tuple[int, str]]:
    world, ava, trace = Line(), Ava(), []
    ava.start(None)
    for _ in range(steps):
        pos = world.pos
        action, _ = ava.act(world.obs())
        trace.append((pos, action.name))
        world.step(action)
    return trace


def test_every_move_is_tried_and_useless_moves_are_never_repeated():
    trace = play(40)
    tried = {(p, a) for p, a in trace if a != "RESET"}
    assert tried == {(p, a) for p in range(4) for a in ("ACTION1", "ACTION2", "ACTION3")}
    for useless in [(p, "ACTION2") for p in range(4)] + [(3, "ACTION1")]:
        assert trace.count(useless) == 1, useless


def test_a_move_that_lost_the_level_is_not_made_again():
    trace = play(40)
    assert trace.count((3, "ACTION3")) == 1


def test_restarts_only_when_nothing_reachable_is_left_to_try():
    trace = play(40)
    first_reset = next(i for i, (_, a) in enumerate(trace) if a == "RESET")
    seen = {(p, a) for p, a in trace[:first_reset]}
    assert seen == {(p, a) for p in range(4) for a in ("ACTION1", "ACTION2", "ACTION3")}
