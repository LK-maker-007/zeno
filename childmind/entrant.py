from __future__ import annotations

from collections import deque
from pathlib import Path

import torch

from childmind.data import encode, prompt
from childmind.infer import answer
from childmind.model import ChildMind, Config

ANSWER_ROOM = 32


class ChildMindEntrant:
    # No memory: only the most recent facts that fit in the context window are visible, as in a plain LLM.
    name = "ours-childmind-nomemory"

    def __init__(self, ckpt: Path, threads: int = 2):
        torch.set_num_threads(threads)
        state = torch.load(ckpt, map_location="cpu")
        self.model = ChildMind(Config(**state["config"]))
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self._recent: deque[str] = deque()

    def tell(self, text: str, source: str, time: int) -> None:
        self._recent.append(text)

    def ask(self, question: str) -> tuple[str, float]:
        budget = self.model.c.ctx - len(encode(prompt([], question))) - ANSWER_ROOM
        facts: list[str] = []
        for t in reversed(self._recent):
            cost = len(encode(f"<f> {t}\n"))
            if cost > budget:
                break
            facts.insert(0, t)
            budget -= cost
        return answer(self.model, prompt(facts, question))
