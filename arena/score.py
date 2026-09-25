from __future__ import annotations

LEVEL_CAP = 1.15
BUDGET_MULTIPLIER = 5


def level_score(human_actions: int, ai_actions: int) -> float:
    # RHAE per completed level (docs.arcprize.org/methodology): squared efficiency ratio, capped at 1.15.
    return min(LEVEL_CAP, (human_actions / ai_actions) ** 2)


def game_score(baseline: tuple[int, ...], actions_per_completed_level: list[int]) -> float:
    # Level-number weights; levels never completed score 0, so an unfinished game cannot reach 100%.
    weights = range(1, len(baseline) + 1)
    earned = sum(
        w * level_score(h, a) for w, h, a in zip(weights, baseline, actions_per_completed_level, strict=False)
    )
    return min(1.0, earned / sum(weights))


def budget(human_actions: int) -> int:
    return BUDGET_MULTIPLIER * human_actions
