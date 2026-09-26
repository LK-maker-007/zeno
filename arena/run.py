from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine.enums import GameAction, GameState

from arena.games import GAMES_DIR, ROOT, Game, load_games, split
from arena.score import budget, game_score
from play.ava import Ava, Explorer


@dataclass
class Obs:
    frame: np.ndarray
    levels_completed: int
    available: tuple[int, ...]
    state: GameState


class Agent(Protocol):
    name: str

    def start(self, game: Game) -> None: ...

    def act(self, obs: Obs) -> tuple[GameAction, dict[str, Any] | None]: ...


class RandomAgent:
    # Floor: uniform over the legal actions, uniform click position. Anything that understands a game must beat it.
    name = "floor-random"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def start(self, game: Game) -> None:
        pass

    def act(self, obs: Obs) -> tuple[GameAction, dict[str, Any] | None]:
        action = GameAction.from_id(self.rng.choice(obs.available))
        if action.is_complex():
            return action, {"x": self.rng.randrange(64), "y": self.rng.randrange(64)}
        return action, None


AGENTS = {"random": RandomAgent, "ava-p0": Explorer, "ava": Ava}


def _obs(f: Any) -> Obs:
    frame = np.asarray(f.frame, dtype=np.int8)
    return Obs(frame[-1], f.levels_completed, tuple(f.available_actions), f.state)


def play(arc: Arcade, game: Game, agent: Agent) -> dict:
    env = arc.make(game.game_id)
    f = env.reset()
    agent.start(game)
    done_levels: list[int] = []
    level_actions = total = 0
    stop = "win"
    while f.state != GameState.WIN and len(done_levels) < len(game.baseline):
        if level_actions >= budget(game.baseline[len(done_levels)]):
            stop = "budget"
            break
        # A lost level is restarted; the restart is a submitted action and is counted as one.
        action, data = (GameAction.RESET, None) if f.state == GameState.GAME_OVER else agent.act(_obs(f))
        nxt = env.step(action, data=data) if data else env.step(action)
        level_actions += 1
        total += 1
        if nxt is None:
            stop = "engine returned no frame"
            print(f"  warning: {game.title} stopped, {stop}", flush=True)
            break
        if nxt.levels_completed < len(done_levels):
            # Level accounting assumes RESET restarts only the current level; stop rather than mis-score.
            stop = "levels completed went down"
            print(f"  warning: {game.title} stopped, {stop}", flush=True)
            break
        if nxt.levels_completed > len(done_levels):
            done_levels.append(level_actions)
            level_actions = 0
        f = nxt
    return {
        "title": game.title,
        "levels": len(game.baseline),
        "completed": len(done_levels),
        "actions_per_level": done_levels,
        "baseline": list(game.baseline),
        "total_actions": total,
        "stop": stop,
        "score": game_score(game.baseline, done_levels),
    }


def run(a: argparse.Namespace) -> Path:
    games = split(load_games(), a.split)
    agent = AGENTS[a.agent](a.seed)
    print(
        f"agent {agent.name}  split {a.split}  games {len(games)}  seed {a.seed}  python {platform.python_version()}",
        flush=True,
    )
    logger = logging.getLogger("arena")
    logger.setLevel(logging.WARNING)
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(GAMES_DIR),
        recordings_dir=str(ROOT / "data" / "arc" / "recordings"),
        logger=logger,
    )
    results, t0 = [], time.perf_counter()
    for g in games:
        r = play(arc, g, agent)
        results.append(r)
        print(
            f"  {r['title']:<6} levels {r['completed']}/{r['levels']}  actions {r['total_actions']:>5}  "
            f"score {100 * r['score']:6.2f}%  stop: {r['stop']}",
            flush=True,
        )
    total = float(np.mean([r["score"] for r in results]))
    print(f"total score {100 * total:.2f}% over {len(results)} games in {time.perf_counter() - t0:.1f}s", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"arena_{agent.name}_{a.split}_seed{a.seed}.json"
    path.write_text(
        json.dumps(
            {
                "command": " ".join(["python -m arena.run", *sys.argv[1:]]),
                "seed": a.seed,
                "split": a.split,
                "agent": agent.name,
                "total_score": total,
                "games": results,
            },
            indent=2,
        )
    )
    print(f"wrote {path}", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=sorted(AGENTS), default="random")
    ap.add_argument("--split", choices=["practice", "heldout", "all"], default="practice")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "arena")
    run(ap.parse_args())
