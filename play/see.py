from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Obj:
    color: int
    cells: tuple[tuple[int, int], ...]

    @property
    def size(self) -> int:
        return len(self.cells)

    def target(self) -> tuple[int, int]:
        # The object's own cell nearest its centre, so a click lands on the object even when it is hollow.
        cy = sum(r for r, _ in self.cells) / self.size
        cx = sum(c for _, c in self.cells) / self.size
        return min(self.cells, key=lambda rc: (rc[0] - cy) ** 2 + (rc[1] - cx) ** 2)


def objects(frame: np.ndarray) -> list[Obj]:
    # Same-colour 4-connected components; the most common colour is the background and is not an object.
    h, w = frame.shape
    bg = int(np.bincount(frame.ravel().astype(np.int64)).argmax())
    seen = np.zeros((h, w), bool)
    out = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or frame[y, x] == bg:
                continue
            color, stack, cells = int(frame[y, x]), [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and frame[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append(Obj(color, tuple(sorted(cells))))
    return out


def place(frame: np.ndarray, ignore: set[int] | None = None) -> bytes:
    # Identity of a situation, leaving out rows that tick regardless of what she does. By default the top and
    # bottom rows: on the practice games a step counter ticks there, which would make every revisit look new.
    rows = {0, frame.shape[0] - 1} if ignore is None else ignore
    return np.delete(frame, sorted(rows), axis=0).tobytes()
