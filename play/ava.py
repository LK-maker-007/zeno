from __future__ import annotations

from collections import deque
from typing import Any

from arcengine.enums import GameAction

from play.see import objects, place

# A move is an action id plus, for a click, its (x, y) cell.
Move = tuple[int, tuple[int, int] | None]
MAX_CLICKS = 32


class Ava:
    # P0: remembers every place she has seen and where each move led, never repeats a move that taught her
    # nothing, and always goes where something is still untried. No model of the game yet: that is P1.
    name = "ava-p0"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def start(self, game: Any) -> None:
        self.level: int | None = None

    def _new_level(self, here: bytes) -> None:
        self.moves: dict[bytes, list[Move]] = {}
        self.led: dict[bytes, dict[Move, bytes]] = {}
        self.bad: set[tuple[bytes, Move]] = set()
        self.start_place, self.prev, self.last = here, None, None

    def _moves(self, obs: Any) -> list[Move]:
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

    def _record(self, here: bytes) -> None:
        if self.prev is None:
            return
        self.led.setdefault(self.prev, {})[self.last] = here
        # Back at the level's start without asking: the move lost the level. Never make it again.
        if here == self.start_place and self.prev != self.start_place:
            self.bad.add((self.prev, self.last))

    def _useful(self, place_: bytes) -> list[tuple[Move, bytes]]:
        return [
            (m, nxt) for m, nxt in self.led.get(place_, {}).items() if nxt != place_ and (place_, m) not in self.bad
        ]

    def _untried(self, place_: bytes) -> list[Move]:
        done = self.led.get(place_, {})
        return [m for m in self.moves.get(place_, []) if m not in done]

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

    def act(self, obs: Any) -> tuple[GameAction, dict[str, Any] | None]:
        here = place(obs.frame)
        if obs.levels_completed != self.level:
            self.level = obs.levels_completed
            self._new_level(here)
        else:
            self._record(here)
        self.moves.setdefault(here, self._moves(obs))
        untried = self._untried(here)
        move = untried[0] if untried else self._path_to_untried(here)
        if move is None:
            # Everything reachable has been tried: start the level again, as a person would.
            self.prev, self.last = None, None
            return GameAction.RESET, None
        self.prev, self.last = here, move
        action, cell = move
        return GameAction.from_id(action), ({"x": cell[0], "y": cell[1]} if cell else None)
