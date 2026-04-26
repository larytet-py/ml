from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .goals import BUILTIN_GOAL_FUNCTIONS, Goal


DEFAULT_GOALS = [
    {"name": "itm_expiries", "kind": "constraint", "direction": "min", "target": 0.0},
    {"name": "max_drawdown_abs", "kind": "constraint", "direction": "min", "target": 5000.0},
    {"name": "avg_pnl", "kind": "objective", "direction": "max", "target": None},
]


@dataclass
class GoalEvaluation:
    values: Dict[str, float]
    violations: Dict[str, float]
    feasible: bool
    objective_score: float


def build_goals(configs: Optional[Iterable[Dict[str, object]]]) -> List[Goal]:
    rows = list(configs or DEFAULT_GOALS)
    if not rows:
        rows = DEFAULT_GOALS
    if len(rows) > 3:
        raise ValueError("At most 3 goals are supported.")

    goals: List[Goal] = []
    for row in rows:
        name = str(row["name"])
        kind = str(row.get("kind", "objective")).strip().lower()
        direction = str(row.get("direction", "max")).strip().lower()
        if kind not in {"constraint", "objective"}:
            raise ValueError(f"Unsupported goal kind '{kind}'")
        if direction not in {"min", "max"}:
            raise ValueError(f"Unsupported goal direction '{direction}'")
        fn = BUILTIN_GOAL_FUNCTIONS.get(name)
        if fn is None:
            raise ValueError(f"Unknown goal name '{name}'. Known goals: {sorted(BUILTIN_GOAL_FUNCTIONS)}")
        target = row.get("target")
        goals.append(
            Goal(
                name=name,
                kind=kind,
                direction=direction,
                value_fn=fn,
                target=None if target is None else float(target),
            )
        )

    objective_count = sum(1 for g in goals if g.kind == "objective")
    if objective_count == 0:
        raise ValueError("At least one objective goal is required.")
    return goals


def evaluate_goals(goals: List[Goal], metrics: Dict[str, float]) -> GoalEvaluation:
    values: Dict[str, float] = {}
    violations: Dict[str, float] = {}
    objective_terms: List[float] = []

    feasible = True
    for goal in goals:
        val = goal.value(metrics)
        values[goal.name] = val
        violation = goal.distance_to_target(metrics)
        violations[goal.name] = violation
        if violation > 0:
            feasible = False
        if goal.kind == "objective":
            objective_terms.append(goal.objective_score(metrics))

    objective_score = float(sum(objective_terms) / max(1, len(objective_terms)))
    return GoalEvaluation(values=values, violations=violations, feasible=feasible, objective_score=objective_score)
