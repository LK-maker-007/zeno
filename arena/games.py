from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GAMES_DIR = ROOT / "data" / "arc" / "environment_files"

# Frozen before any agent was written: these 12 public games are never used to develop or tune an agent.
# Chosen by sorting the 25 titles and taking every other one, starting from the second.
HELD_OUT = frozenset({"BP35", "CN04", "FT09", "KA59", "LP85", "M0R0", "RE86", "SB26", "SK48", "SU15", "TR87", "VC33"})


@dataclass(frozen=True)
class Game:
    game_id: str
    title: str
    tags: tuple[str, ...]
    baseline: tuple[int, ...]

    @property
    def held_out(self) -> bool:
        return self.title in HELD_OUT


def load_games(games_dir: Path = GAMES_DIR) -> list[Game]:
    games = []
    for meta in sorted(games_dir.glob("*/*/metadata.json")):
        m = json.loads(meta.read_text())
        games.append(Game(m["game_id"], m["title"], tuple(m.get("tags", [])), tuple(m["baseline_actions"])))
    return sorted(games, key=lambda g: g.title)


def split(games: list[Game], name: str) -> list[Game]:
    if name == "all":
        return games
    return [g for g in games if g.held_out == (name == "heldout")]
