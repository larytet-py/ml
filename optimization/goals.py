from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional


MetricFn = Callable[[Dict[str, float]], float]


def goal_itm(metrics: Dict[str, float]) -> float:
    return float(metrics.get("itm_expiries", 0.0))


def goal_drawdown(metrics: Dict[str, float]) -> float:
    return abs(float(metrics.get("max_drawdown", 0.0)))


def goal_avg_pnl(metrics: Dict[str, float]) -> float:
    return float(metrics.get("avg_pnl", 0.0))


BUILTIN_GOAL_FUNCTIONS: Dict[str, MetricFn] = {
    "itm_expiries": goal_itm,
    "max_drawdown_abs": goal_drawdown,
    "avg_pnl": goal_avg_pnl,
    "total_pnl": lambda m: float(m.get("total_pnl", 0.0)),
    "win_rate": lambda m: float(m.get("win_rate", 0.0)),
}


@dataclass(frozen=True)
class Goal:
    name: str
    direction: str  # min|max
    kind: str  # constraint|objective
    value_fn: MetricFn
    target: Optional[float] = None

    def value(self, metrics: Dict[str, float]) -> float:
        return float(self.value_fn(metrics))

    def is_satisfied(self, metrics: Dict[str, float]) -> bool:
        if self.kind != "constraint" or self.target is None:
            return True
        val = self.value(metrics)
        if self.direction == "min":
            return val <= self.target
        if self.direction == "max":
            return val >= self.target
        raise ValueError(f"Unsupported direction '{self.direction}'")

    def distance_to_target(self, metrics: Dict[str, float]) -> float:
        if self.kind != "constraint" or self.target is None:
            return 0.0
        val = self.value(metrics)
        if self.direction == "min":
            return max(0.0, val - self.target)
        if self.direction == "max":
            return max(0.0, self.target - val)
        raise ValueError(f"Unsupported direction '{self.direction}'")

    def objective_score(self, metrics: Dict[str, float]) -> float:
        val = self.value(metrics)
        if self.direction == "max":
            return val
        if self.direction == "min":
            return -val
        raise ValueError(f"Unsupported direction '{self.direction}'")
