from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

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

from play.ava import Ava, Explorer  # noqa: E402


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
    world, ava, trace = Line(), Explorer(), []
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


MAZE_SHIFT = {
    GameAction.ACTION1: (-2, 0),
    GameAction.ACTION2: (2, 0),
    GameAction.ACTION3: (0, -2),
    GameAction.ACTION4: (0, 2),
}


class Maze:
    # Toy key game: a 2x2 two-colour body on a floor walled by the most common colour, moving 2 cells per key.
    # Touching the 2x2 target ends the level, and the next level has the same layout. Row 63 ticks every step.
    def __init__(self):
        self.level, self.steps, self.pos = 0, 0, (30, 10)

    def obs(self) -> Obs:
        f = np.full((64, 64), 4, np.int8)
        f[20:42, 8:40] = 3
        f[20:34, 20:22] = 4
        f[30:32, 34:36] = 8
        r, c = self.pos
        f[r, c : c + 2] = 12
        f[r + 1, c : c + 2] = 9
        f[63, : self.steps % 64] = 11
        return Obs(f, self.level, (1, 2, 3, 4))

    def step(self, action: GameAction) -> None:
        self.steps += 1
        dy, dx = MAZE_SHIFT.get(action, (0, 0))
        r, c = self.pos[0] + dy, self.pos[1] + dx
        f = self.obs().frame
        if all(f[rr, cc] in (3, 8, 9, 12) for rr in (r, r + 1) for cc in (c, c + 1)):
            self.pos = (r, c)
        if abs(self.pos[0] - 30) <= 2 and abs(self.pos[1] - 34) <= 2:
            self.level, self.pos = self.level + 1, (30, 10)


def test_p1_finds_its_body_and_walks_to_the_target_then_goes_straight_there_next_level():
    world, ava, per_level, n = Maze(), Ava(), [], 0
    ava.start(None)
    while world.level < 2 and n < 200:
        level = world.level
        action, _ = ava.act(world.obs())
        world.step(action)
        n += 1
        if world.level != level:
            per_level.append(n)
            n = 0
    assert world.level == 2, per_level
    assert ava.body.sprite is not None and 4 in ava.body.walls and 3 in ava.body.floors
    # The wall at columns 20-21 forces a detour; the second level needs no probing and no dead ends.
    assert per_level[1] <= per_level[0]
    assert per_level[1] <= 16


def test_never_restarts_at_a_level_start_or_right_after_a_restart():
    # A restart as a level's first action, or straight after another restart, throws the whole game back to
    # level 0 in the ARC-AGI-3 engine. A world with no moves at all is where a careless explorer would do it.
    ex = Explorer()
    ex.start(None)
    frame = np.zeros((64, 64), np.int8)
    trace = [ex.act(Obs(frame, 0, (1,)))[0] for _ in range(6)]
    for prev, cur in pairwise(trace):
        assert not (prev == GameAction.RESET and cur == GameAction.RESET)
    assert trace[0] != GameAction.RESET
