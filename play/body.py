from __future__ import annotations

from collections import deque

import numpy as np

from play.see import Obj, objects

Shift = tuple[int, int]
REACH = 10
MAX_OBJECT = 400


def rigid_move(before: np.ndarray, after: np.ndarray, bg: int) -> tuple[Shift, dict[tuple[int, int], int]] | None:
    # The displacement that explains the most changed cells as non-background content carried over from `before`.
    # Returns the shift and the moved sprite as {offset from its top-left: colour}, or None if nothing moved.
    ys, xs = np.nonzero(before != after)
    if len(ys) == 0:
        return None
    best, best_n = None, 0
    for dy in range(-REACH, REACH + 1):
        for dx in range(-REACH, REACH + 1):
            if (dy, dx) == (0, 0):
                continue
            sy, sx = ys - dy, xs - dx
            ok = (sy >= 0) & (sy < before.shape[0]) & (sx >= 0) & (sx < before.shape[1])
            src = before[sy[ok], sx[ok]]
            n = int(((after[ys[ok], xs[ok]] == src) & (src != bg)).sum())
            if n > best_n:
                best, best_n = (dy, dx), n
    if best is None or best_n < 3:
        return None
    dy, dx = best
    sy, sx = ys - dy, xs - dx
    ok = (sy >= 0) & (sy < before.shape[0]) & (sx >= 0) & (sx < before.shape[1])
    src = before[sy[ok], sx[ok]]
    hit = (after[ys[ok], xs[ok]] == src) & (src != bg)
    cells = largest_cluster(set(zip(ys[ok][hit].tolist(), xs[ok][hit].tolist(), strict=True)))
    if len(cells) < 4:
        return None
    top, left = min(r for r, _ in cells), min(c for _, c in cells)
    sprite = {(r - top, c - left): int(after[r, c]) for r, c in cells}
    return best, sprite


def largest_cluster(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    # Coincidental matches elsewhere on the screen are scattered; the body is one 8-connected piece.
    best: list[tuple[int, int]] = []
    left = set(cells)
    while left:
        stack, part = [left.pop()], []
        while stack:
            r, c = stack.pop()
            part.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (r + dr, c + dc) in left:
                        left.remove((r + dr, c + dc))
                        stack.append((r + dr, c + dc))
        if len(part) > len(best):
            best = part
    return best


def find(
    frame: np.ndarray, sprite: dict[tuple[int, int], int], near: tuple[int, int] | None
) -> tuple[int, int] | None:
    # Top-left positions where the whole sprite appears, nearest to `near` first.
    h, w = frame.shape
    sh = max(r for r, _ in sprite) + 1
    sw = max(c for _, c in sprite) + 1
    anchor, colour = next(iter(sprite.items()))
    cand = np.argwhere(frame == colour) - np.array(anchor)
    hits = []
    for top, left in cand.tolist():
        if (
            0 <= top <= h - sh
            and 0 <= left <= w - sw
            and all(frame[top + r, left + c] == v for (r, c), v in sprite.items())
        ):
            hits.append((top, left))
    if hits:
        if near is None:
            return hits[0]
        return min(hits, key=lambda p: abs(p[0] - near[0]) + abs(p[1] - near[1]))
    if near is None:
        return None
    # Not found exactly: the body may have changed a little (turned, carries something). Take the best partial
    # match near where it should be, if most of it is there.
    best, best_n = None, 0
    for top in range(max(0, near[0] - REACH), min(h - sh, near[0] + REACH) + 1):
        for left in range(max(0, near[1] - REACH), min(w - sw, near[1] + REACH) + 1):
            n = sum(frame[top + r, left + c] == v for (r, c), v in sprite.items())
            if n > best_n:
                best, best_n = (top, left), n
    return best if best_n >= 0.75 * len(sprite) else None


class Body:
    # What Ava has learned about the thing she controls in one game: how each key moves it, what it can stand on,
    # what stops it and what kills it. Kept across levels, as a person keeps it.
    def __init__(self):
        self.shifts: dict[int, Shift] = {}
        self.dead_keys: set[int] = set()
        self.sprite: dict[tuple[int, int], int] | None = None
        self.walls: set[int] = set()
        self.floors: set[int] = set()
        self.hazards: set[int] = set()
        self.goals: set[int] = set()

    def footprint(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        return [(pos[0] + r, pos[1] + c) for r, c in self.sprite]

    def passable(
        self, frame: np.ndarray, pos: tuple[int, int], own: set[tuple[int, int]], avoid: frozenset = frozenset()
    ) -> bool:
        h, w = frame.shape
        for r, c in self.footprint(pos):
            if not (0 <= r < h and 0 <= c < w) or (r, c) in avoid:
                return False
            if (r, c) in own:
                continue
            v = int(frame[r, c])
            if v in self.walls or v in self.hazards:
                return False
        return True

    def learn_block(self, frame: np.ndarray, pos: tuple[int, int], shift: Shift, own: set[tuple[int, int]]) -> None:
        # The move was refused: whatever colour stood where the body would have gone is a wall.
        for r, c in self.footprint((pos[0] + shift[0], pos[1] + shift[1])):
            if 0 <= r < frame.shape[0] and 0 <= c < frame.shape[1] and (r, c) not in own:
                v = int(frame[r, c])
                if v not in self.floors:
                    self.walls.add(v)

    def learn_floor(self, before: np.ndarray, after: np.ndarray, old: tuple[int, int], new: tuple[int, int]) -> None:
        # Cells the body just left show what it was standing on.
        now = set(self.footprint(new))
        for r, c in self.footprint(old):
            if (r, c) not in now:
                v = int(after[r, c])
                self.floors.add(v)
                self.walls.discard(v)

    def targets(self, frame: np.ndarray, own: set[tuple[int, int]], bg: int, touched: set) -> list[Obj]:
        out = []
        for o in objects(frame):
            if o.color == bg:
                continue
            if o.size > MAX_OBJECT or own.intersection(o.cells) or (o.color, o.cells[0]) in touched:
                continue
            out.append(o)
        return out

    def route(
        self, frame: np.ndarray, pos: tuple[int, int], goals: list[Obj], avoid: frozenset = frozenset()
    ) -> tuple[list[int], Obj] | None:
        # Shortest key sequence that brings the body onto or next to a goal object without crossing `avoid`.
        own = set(self.footprint(pos))
        cells = {}
        for o in goals:
            for r, c in o.cells:
                for rr, cc in ((r, c), (r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    cells.setdefault((rr, cc), o)
        prev: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {pos: None}
        queue = deque([pos])
        while queue:
            p = queue.popleft()
            hit = next((cells[x] for x in self.footprint(p) if x in cells and x not in own), None)
            if hit is not None and p != pos:
                keys = []
                while prev[p] is not None:
                    p, k = prev[p]
                    keys.append(k)
                return keys[::-1], hit
            for key, (dy, dx) in self.shifts.items():
                q = (p[0] + dy, p[1] + dx)
                if q not in prev and self.passable(frame, q, own, avoid):
                    prev[q] = (p, key)
                    queue.append(q)
        return None
