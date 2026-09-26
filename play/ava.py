from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from arcengine.enums import GameAction

from play.body import Body, find, rigid_move
from play.see import Obj, objects, place

# A move is an action id plus, for a click, its (x, y) cell.
Move = tuple[int, tuple[int, int] | None]
MAX_CLICKS = 32


def moves(obs: Any) -> list[Move]:
    keys = [(a, None) for a in obs.available if not GameAction.from_id(a).is_complex()]
    clicks = []
    if any(GameAction.from_id(a).is_complex() for a in obs.available):
        # One click per distinct-looking object, smallest first: buttons and switches tend to be small.
        seen, click = set(), next(a for a in obs.available if GameAction.from_id(a).is_complex())
        for o in sorted(objects(obs.frame), key=lambda o: (o.size, o.color)):
            look = (o.color, o.size)
            if look not in seen:
                seen.add(look)
                y, x = o.target()
                clicks.append((click, (x, y)))
    return keys + clicks[:MAX_CLICKS]


def to_action(move: Move) -> tuple[GameAction, dict[str, Any] | None]:
    action, cell = move
    return GameAction.from_id(action), ({"x": cell[0], "y": cell[1]} if cell else None)


class Explorer:
    # P0: remembers every place she has seen and where each move led, never repeats a move that taught her
    # nothing, and always goes where something is still untried.
    name = "ava-p0"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.ignore: set[int] | None = None

    def start(self, game: Any) -> None:
        self.level: int | None = None

    def _new_level(self, here: bytes) -> None:
        self.moves: dict[bytes, list[Move]] = {}
        self.led: dict[bytes, dict[Move, bytes]] = {}
        self.bad: set[tuple[bytes, Move]] = set()
        self.start_place, self.prev, self.last = here, None, None
        self.fresh = True

    def see(self, obs: Any) -> bytes:
        here = place(obs.frame, self.ignore)
        if obs.levels_completed != self.level:
            self.level = obs.levels_completed
            self._new_level(here)
        elif self.prev is not None:
            self.led.setdefault(self.prev, {})[self.last] = here
            # Back at the level's start without asking: the move lost the level. Never make it again.
            if here == self.start_place and self.prev != self.start_place:
                self.bad.add((self.prev, self.last))
        if here == self.start_place:
            self.fresh = True
        self.moves.setdefault(here, moves(obs))
        return here

    def took(self, here: bytes | None, move: Move | None) -> None:
        self.prev, self.last = here, move

    def _useful(self, p: bytes) -> list[tuple[Move, bytes]]:
        return [(m, nxt) for m, nxt in self.led.get(p, {}).items() if nxt != p and (p, m) not in self.bad]

    def _untried(self, p: bytes) -> list[Move]:
        done = self.led.get(p, {})
        return [m for m in self.moves.get(p, []) if m not in done]

    def _path_to_untried(self, here: bytes) -> Move | None:
        # Breadth-first over remembered moves to the nearest place that still has something to try.
        first: dict[bytes, Move | None] = {here: None}
        queue = deque([here])
        while queue:
            p = queue.popleft()
            if p != here and self._untried(p):
                return first[p]
            for m, nxt in self._useful(p):
                if nxt not in first:
                    first[nxt] = m if p == here else first[p]
                    queue.append(nxt)
        return None

    def choose(self, here: bytes) -> Move | None:
        untried = self._untried(here)
        return untried[0] if untried else self._path_to_untried(here)

    def decide(self, here: bytes) -> Move | None:
        # None means restart the level. Never at a level's start or straight after a restart: the engine then
        # restarts the whole game from level 0 (arcengine base_game.handle_reset, _action_count == 0).
        move = self.choose(here)
        if move is None and self.fresh:
            move = next(iter(self.moves.get(here, [])), None)
        self.fresh = move is None
        return move

    def act(self, obs: Any) -> tuple[GameAction, dict[str, Any] | None]:
        here = self.see(obs)
        move = self.decide(here)
        if move is None:
            # Everything reachable has been tried: start the level again, as a person would.
            self.took(None, None)
            return GameAction.RESET, None
        self.took(here, move)
        return to_action(move)


class Ava:
    # P1: finds what she controls by pressing each key once, learns walls, floors and hazards from what happens,
    # and walks the shortest route to the nearest object she has not touched. Knowledge of the body carries across
    # levels. Without a body (click-only games, or nothing moved) she explores like P0.
    name = "ava-p1"

    def __init__(self, seed: int = 0):
        self.explorer = Explorer(seed)

    def start(self, game: Any) -> None:
        self.explorer.start(game)
        self.body = Body()
        self.level: int | None = None
        self.before: np.ndarray | None = None
        self.key: int | None = None
        self.churn, self.watched, self.hud = np.zeros(64, int), 0, {0, 63}

    def _new_level(self, frame: np.ndarray) -> None:
        self.touched: set = set()
        self.plan: list[int] = []
        self.target: Obj | None = None
        self.arriving: Obj | None = None
        self.switches: list[Obj] = []
        self.pos = find(frame, self.body.sprite, None) if self.body.sprite else None
        self.start_pos = self.pos

    def _own(self) -> set[tuple[int, int]]:
        return set(self.body.footprint(self.pos)) if self.pos is not None and self.body.sprite else set()

    def _learn(self, frame: np.ndarray, bg: int) -> None:
        # What the last key press did.
        key, before, body = self.key, self.before, self.body
        if key is None or before is None:
            return
        if key not in body.shifts and key not in body.dead_keys:
            mv = rigid_move(before, frame, bg)
            if mv is None:
                body.dead_keys.add(key)
                return
            shift, sprite = mv
            body.shifts[key] = shift
            if body.sprite is None:
                body.sprite = sprite
                self.pos = find(frame, sprite, None)
                self.start_pos = find(before, sprite, None)
            return
        if self.pos is None or body.sprite is None:
            self.pos = find(frame, body.sprite, None) if body.sprite else None
            return
        shift = body.shifts.get(key)
        if shift is None:
            return
        own = self._own()
        expected = (self.pos[0] + shift[0], self.pos[1] + shift[1])
        now = find(frame, body.sprite, expected)
        if now == expected:
            body.learn_floor(before, frame, self.pos, now)
        elif now == self.pos:
            body.learn_block(before, self.pos, shift, own)
            self.plan = []
        elif self.reset_seen:
            # The whole screen went back to the level's start: the step lost the level, so whatever the body
            # stepped onto is dangerous.
            for r, c in body.footprint(expected):
                if 0 <= r < before.shape[0] and 0 <= c < before.shape[1] and (r, c) not in own:
                    v = int(before[r, c])
                    if v not in body.floors:
                        body.hazards.add(v)
            self.plan = []
        else:
            self.plan = []
        self.pos = now

    def _watch(self, before: np.ndarray, frame: np.ndarray, old: tuple | None) -> bool:
        # Rows that change on nearly every step, apart from her own body, are a counter or an energy bar, not
        # the world: she stops counting them as part of a place. Returns whether the world itself changed.
        h, w = frame.shape
        d = before != frame
        if self.body.sprite is not None:
            if old is None or self.pos is None:
                return False
            for p in (old, self.pos):
                for r, c in self.body.footprint(p):
                    if 0 <= r < h and 0 <= c < w:
                        d[r, c] = False
        self.churn[np.nonzero(d.any(1))[0]] += 1
        self.watched += 1
        hud = {0, h - 1}
        if self.watched >= 4:
            hud |= {int(r) for r in np.nonzero(self.churn >= 0.8 * self.watched)[0]}
        if hud != self.hud:
            self.hud = self.explorer.ignore = hud
            self.explorer._new_level(place(frame, hud))
        d[sorted(hud)] = False
        return bool(d.any())

    def _next_key(self, frame: np.ndarray, bg: int, keys: list[int]) -> int | None:
        body = self.body
        unprobed = [k for k in keys if k not in body.shifts and k not in body.dead_keys]
        if unprobed:
            return unprobed[0]
        if body.sprite is None or self.pos is None or not body.shifts:
            return None
        if not self.plan:
            goals = body.targets(frame, self._own(), bg, self.touched)
            preferred = [o for o in goals if o.color in body.goals]
            avoid = frozenset(c for s in self.switches for c in s.cells)
            found = (body.route(frame, self.pos, preferred, avoid) if preferred else None) or body.route(
                frame, self.pos, goals, avoid
            )
            if found is None:
                return None
            self.plan, self.target = found
        key = self.plan.pop(0)
        if not self.plan and self.target is not None:
            self.touched.add((self.target.color, self.target.cells[0]))
            self.arriving = self.target
        return key

    def act(self, obs: Any) -> tuple[GameAction, dict[str, Any] | None]:
        frame = np.asarray(obs.frame, dtype=np.int16)
        bg = int(np.bincount(frame.ravel()).argmax())
        here = self.explorer.see(obs)
        self.reset_seen = here == self.explorer.start_place and self.explorer.prev not in (None, here)
        if obs.levels_completed != self.level:
            if self.level is not None and self.target is not None:
                # The last thing she went for finished the level: go for that kind of thing first next time.
                self.body.goals.add(self.target.color)
            self.level = obs.levels_completed
            self._new_level(frame)
        else:
            old = self.pos
            self._learn(frame, bg)
            if self.before is not None and self._watch(self.before, frame, old) and self.arriving is not None:
                # Reaching that thing changed the world. It is a switch: keep off it from now on, and look again
                # at everything else, since the change may have made something else work.
                self.switches.append(self.arriving)
                self.touched = {(s.color, s.cells[0]) for s in self.switches}
                self.plan = []
            self.arriving = None
        keys = [a for a in obs.available if not GameAction.from_id(a).is_complex()]
        key = self._next_key(frame, bg, keys)
        self.before = frame
        if key is not None:
            self.key = key
            self.explorer.fresh = False
            self.explorer.took(here, (key, None))
            return GameAction.from_id(key), None
        self.key = None
        move = self.explorer.decide(here)
        if move is None:
            self.explorer.took(None, None)
            self.before = None
            return GameAction.RESET, None
        self.explorer.took(here, move)
        if move[1] is None:
            self.key = move[0]
        return to_action(move)
