from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.neighbors import NearestNeighbors

from .goal_registry import GoalEvaluation, evaluate_goals
from .goals import Goal


@dataclass
class ParamSpec:
    low: float
    high: float
    is_int: bool = False


@dataclass
class TrialResult:
    trial_id: int
    params: Dict[str, float]
    metrics: Dict[str, float]
    goals: GoalEvaluation
    penalized_score: float


class ConstrainedBayesianOptimizer:
    def __init__(
        self,
        search_space: Dict[str, ParamSpec],
        goals: List[Goal],
        evaluator: Callable[[Dict[str, float]], Dict[str, float]],
        trials_parquet: str,
        eval_cache_parquet: Optional[str] = None,
        seed: int = 42,
        n_random_init: int = 20,
        candidate_pool_size: int = 512,
        constraint_penalty: float = 1000.0,
        progress_interval_seconds: float = 5.0,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.search_space = search_space
        self.goals = goals
        self.evaluator = evaluator
        self.trials_parquet = Path(trials_parquet)
        self.eval_cache_parquet = Path(eval_cache_parquet) if eval_cache_parquet else None
        self.seed = int(seed)
        self.n_random_init = int(n_random_init)
        self.candidate_pool_size = int(candidate_pool_size)
        self.constraint_penalty = float(constraint_penalty)
        self.progress_interval_seconds = float(progress_interval_seconds)
        self.logger = logger or (lambda message: print(message, flush=True))

        self.rng = np.random.default_rng(self.seed)
        self.param_names = list(search_space)
        self._trial_rows: List[Dict[str, object]] = []
        self._eval_cache: Dict[str, Dict[str, float]] = {}
        self._cache_dirty = False
        self._load_eval_cache()

    def _log(self, message: str) -> None:
        self.logger(message)

    def _normalize(self, params: Dict[str, float]) -> np.ndarray:
        out: List[float] = []
        for name in self.param_names:
            spec = self.search_space[name]
            width = spec.high - spec.low
            if width <= 0:
                out.append(0.0)
            else:
                out.append((float(params[name]) - spec.low) / width)
        return np.asarray(out, dtype=float)

    def _denormalize(self, x: np.ndarray) -> Dict[str, float]:
        params: Dict[str, float] = {}
        for i, name in enumerate(self.param_names):
            spec = self.search_space[name]
            val = spec.low + float(x[i]) * (spec.high - spec.low)
            if spec.is_int:
                val = int(round(val))
                val = int(max(spec.low, min(spec.high, val)))
            params[name] = float(val)
        return params

    def _random_candidate(self) -> Dict[str, float]:
        x = self.rng.uniform(0.0, 1.0, len(self.param_names))
        return self._denormalize(x)

    def _penalized(self, objective_score: float, violations: Dict[str, float]) -> float:
        total_violation = float(sum(max(0.0, v) for v in violations.values()))
        return objective_score - self.constraint_penalty * total_violation

    def _fit_gp(self, trials: List[TrialResult]) -> Optional[GaussianProcessRegressor]:
        if len(trials) < max(5, len(self.param_names) + 1):
            return None
        X = np.vstack([self._normalize(t.params) for t in trials])
        y = np.asarray([t.penalized_score for t in trials], dtype=float)

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-4)
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=self.seed,
        )
        model.fit(X, y)
        return model

    def _propose_via_gp(self, model: GaussianProcessRegressor) -> Dict[str, float]:
        X_cand = self.rng.uniform(0.0, 1.0, size=(self.candidate_pool_size, len(self.param_names)))
        mu, std = model.predict(X_cand, return_std=True)
        # UCB with gentle exploration.
        ucb = mu + 1.5 * std
        best_idx = int(np.argmax(ucb))
        return self._denormalize(X_cand[best_idx])

    def _trials_header(self) -> List[str]:
        columns = ["trial_id", "penalized_score", "feasible", "objective_score"]
        columns.extend(f"param__{name}" for name in self.param_names)
        metric_cols = sorted(
            {
                "total",
                "wins",
                "win_rate",
                "itm_expiries",
                "itm_rate",
                "total_pnl",
                "avg_pnl",
                "median_pnl",
                "avg_return_on_spot",
                "max_drawdown",
            }
        )
        columns.extend(f"metric__{name}" for name in metric_cols)
        columns.extend(f"goal__{goal.name}" for goal in self.goals)
        columns.extend(f"viol__{goal.name}" for goal in self.goals)
        return columns

    def _trial_to_row(self, trial: TrialResult) -> Dict[str, object]:
        row: Dict[str, object] = {
            "trial_id": trial.trial_id,
            "penalized_score": trial.penalized_score,
            "feasible": int(trial.goals.feasible),
            "objective_score": trial.goals.objective_score,
        }
        row.update({f"param__{k}": v for k, v in trial.params.items()})
        for k in {"total", "wins", "win_rate", "itm_expiries", "itm_rate", "total_pnl", "avg_pnl", "median_pnl", "avg_return_on_spot", "max_drawdown"}:
            row[f"metric__{k}"] = trial.metrics.get(k)
        for goal in self.goals:
            row[f"goal__{goal.name}"] = trial.goals.values.get(goal.name)
            row[f"viol__{goal.name}"] = trial.goals.violations.get(goal.name)
        return row

    def _persist_trials(self) -> None:
        self.trials_parquet.parent.mkdir(parents=True, exist_ok=True)
        header = self._trials_header()
        df = pd.DataFrame(self._trial_rows)
        for col in header:
            if col not in df.columns:
                df[col] = np.nan
        df = df[header]
        df.to_parquet(self.trials_parquet, index=False)

    def _persist_eval_cache(self) -> None:
        if self.eval_cache_parquet is None:
            return
        self.eval_cache_parquet.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for key, metrics in self._eval_cache.items():
            row = {"cache_key": key}
            for metric_name in [
                "total",
                "wins",
                "win_rate",
                "itm_expiries",
                "itm_rate",
                "total_pnl",
                "avg_pnl",
                "median_pnl",
                "avg_return_on_spot",
                "max_drawdown",
            ]:
                row[f"metric__{metric_name}"] = metrics.get(metric_name)
            rows.append(row)
        pd.DataFrame(rows).to_parquet(self.eval_cache_parquet, index=False)
        self._cache_dirty = False

    def _load_eval_cache(self) -> None:
        if self.eval_cache_parquet is None or not self.eval_cache_parquet.exists():
            return
        df = pd.read_parquet(self.eval_cache_parquet)
        out: Dict[str, Dict[str, float]] = {}
        for _, row in df.iterrows():
            key = str(row.get("cache_key", ""))
            if not key:
                continue
            out[key] = {
                "total": float(row.get("metric__total", 0.0)),
                "wins": float(row.get("metric__wins", 0.0)),
                "win_rate": float(row.get("metric__win_rate", 0.0)),
                "itm_expiries": float(row.get("metric__itm_expiries", 0.0)),
                "itm_rate": float(row.get("metric__itm_rate", 0.0)),
                "total_pnl": float(row.get("metric__total_pnl", 0.0)),
                "avg_pnl": float(row.get("metric__avg_pnl", 0.0)),
                "median_pnl": float(row.get("metric__median_pnl", 0.0)),
                "avg_return_on_spot": float(row.get("metric__avg_return_on_spot", 0.0)),
                "max_drawdown": float(row.get("metric__max_drawdown", 0.0)),
            }
        self._eval_cache = out

    def _params_cache_key(self, params: Dict[str, float]) -> str:
        stable: Dict[str, float] = {}
        for name in self.param_names:
            spec = self.search_space[name]
            value = float(params[name])
            if spec.is_int:
                stable[name] = float(int(round(value)))
            else:
                stable[name] = float(round(value, 12))
        return json.dumps(stable, sort_keys=True)

    def load_existing_trials(self) -> List[TrialResult]:
        self._trial_rows = []
        if not self.trials_parquet.exists():
            return []
        df = pd.read_parquet(self.trials_parquet)
        trials: List[TrialResult] = []
        for _, row in df.iterrows():
            params = {name: float(row[f"param__{name}"]) for name in self.param_names if f"param__{name}" in row}
            metrics = {
                "total": float(row.get("metric__total", 0.0)),
                "wins": float(row.get("metric__wins", 0.0)),
                "win_rate": float(row.get("metric__win_rate", 0.0)),
                "itm_expiries": float(row.get("metric__itm_expiries", 0.0)),
                "itm_rate": float(row.get("metric__itm_rate", 0.0)),
                "total_pnl": float(row.get("metric__total_pnl", 0.0)),
                "avg_pnl": float(row.get("metric__avg_pnl", 0.0)),
                "median_pnl": float(row.get("metric__median_pnl", 0.0)),
                "avg_return_on_spot": float(row.get("metric__avg_return_on_spot", 0.0)),
                "max_drawdown": float(row.get("metric__max_drawdown", 0.0)),
            }
            goals_eval = evaluate_goals(self.goals, metrics)
            trials.append(
                TrialResult(
                    trial_id=int(row.get("trial_id", len(trials))),
                    params=params,
                    metrics=metrics,
                    goals=goals_eval,
                    penalized_score=float(row.get("penalized_score", 0.0)),
                )
            )
            self._trial_rows.append(self._trial_to_row(trials[-1]))
        return trials

    def run(self, n_trials: int) -> List[TrialResult]:
        trials = self.load_existing_trials()
        start_id = len(trials)
        if not self._eval_cache:
            for trial in trials:
                self._eval_cache[self._params_cache_key(trial.params)] = dict(trial.metrics)
        start_time = time.monotonic()
        last_progress_time = start_time
        cache_hits = 0
        feasible = [t for t in trials if t.goals.feasible]
        best_feasible = max(feasible, key=lambda t: t.goals.objective_score) if feasible else None

        for i in range(start_id, n_trials):
            model = self._fit_gp(trials)
            phase = "random"
            if model is None or i < self.n_random_init:
                params = self._random_candidate()
            else:
                phase = "gp"
                params = self._propose_via_gp(model)

            cache_key = self._params_cache_key(params)
            if cache_key in self._eval_cache:
                metrics = dict(self._eval_cache[cache_key])
                cache_hits += 1
            else:
                metrics = self.evaluator(params)
                self._eval_cache[cache_key] = dict(metrics)
                self._cache_dirty = True
            goals_eval = evaluate_goals(self.goals, metrics)
            penalized = self._penalized(goals_eval.objective_score, goals_eval.violations)
            trial = TrialResult(
                trial_id=i,
                params=params,
                metrics=metrics,
                goals=goals_eval,
                penalized_score=penalized,
            )
            trials.append(trial)
            self._trial_rows.append(self._trial_to_row(trial))
            self._persist_trials()
            if self._cache_dirty:
                self._persist_eval_cache()

            if trial.goals.feasible and (best_feasible is None or trial.goals.objective_score > best_feasible.goals.objective_score):
                best_feasible = trial
                self._log(
                    "[bo] improvement "
                    f"trial={i + 1}/{n_trials} objective={trial.goals.objective_score:.6f} "
                    f"avg_pnl={trial.metrics.get('avg_pnl', float('nan')):.2f} "
                    f"itm={trial.metrics.get('itm_expiries', float('nan')):.1f} "
                    f"max_drawdown={trial.metrics.get('max_drawdown', float('nan')):.2f}"
                )

            if self.progress_interval_seconds > 0:
                now = time.monotonic()
                if now - last_progress_time >= self.progress_interval_seconds:
                    elapsed = now - start_time
                    feasible_count = sum(1 for t in trials if t.goals.feasible)
                    if best_feasible is None:
                        self._log(
                            "[bo] progress "
                            f"t+{elapsed:.1f}s trial={i + 1}/{n_trials} phase={phase} "
                            f"feasible=0 cache_hits={cache_hits}"
                        )
                    else:
                        self._log(
                            "[bo] progress "
                            f"t+{elapsed:.1f}s trial={i + 1}/{n_trials} phase={phase} "
                            f"feasible={feasible_count} cache_hits={cache_hits} "
                            f"best_objective={best_feasible.goals.objective_score:.6f} "
                            f"best_avg_pnl={best_feasible.metrics.get('avg_pnl', float('nan')):.2f} "
                            f"best_itm={best_feasible.metrics.get('itm_expiries', float('nan')):.1f} "
                            f"best_max_drawdown={best_feasible.metrics.get('max_drawdown', float('nan')):.2f}"
                        )
                    last_progress_time = now
        if self._cache_dirty:
            self._persist_eval_cache()
        return trials


def extract_region_summary(
    trials: Iterable[TrialResult],
    search_space: Dict[str, ParamSpec],
    performance_tolerance_pct: float,
    min_component_fraction: float,
) -> Dict[str, object]:
    trial_list = list(trials)
    feasible = [t for t in trial_list if t.goals.feasible]
    if not feasible:
        return {
            "selected_component": None,
            "components": [],
            "note": "No feasible trials found.",
        }

    best_obj = max(t.goals.objective_score for t in feasible)
    tolerance = abs(best_obj) * (performance_tolerance_pct / 100.0)
    kept = [t for t in feasible if t.goals.objective_score >= best_obj - tolerance]
    if not kept:
        kept = feasible

    param_names = list(search_space)
    X = []
    for t in kept:
        vec = []
        for name in param_names:
            spec = search_space[name]
            width = spec.high - spec.low
            if width <= 0:
                vec.append(0.0)
            else:
                vec.append((float(t.params[name]) - spec.low) / width)
        X.append(vec)
    X_arr = np.asarray(X, dtype=float)

    if len(kept) == 1:
        components = [np.array([0], dtype=int)]
    else:
        k = min(10, len(kept) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_arr)
        indices = nbrs.kneighbors(X_arr, return_distance=False)
        graph = {i: set() for i in range(len(kept))}
        for i in range(len(kept)):
            for j in indices[i][1:]:
                graph[i].add(int(j))
                graph[int(j)].add(i)

        visited = set()
        components = []
        for i in range(len(kept)):
            if i in visited:
                continue
            stack = [i]
            comp = []
            visited.add(i)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nxt in graph[cur]:
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            components.append(np.asarray(sorted(comp), dtype=int))

    min_size = max(1, int(np.ceil(len(kept) * min_component_fraction)))
    rows = []
    for comp_id, comp_idx in enumerate(components):
        if len(comp_idx) < min_size:
            continue
        comp_X = X_arr[comp_idx]
        comp_trials = [kept[int(i)] for i in comp_idx]
        widths = comp_X.max(axis=0) - comp_X.min(axis=0)
        local_width_by_dim = {name: float(widths[idx]) for idx, name in enumerate(param_names)}
        active_dim_count = int(np.sum(widths < 0.15))
        flat_volume_proxy = float(np.prod(np.clip(widths, 1e-12, None)))
        objective_values = np.asarray([t.goals.objective_score for t in comp_trials], dtype=float)

        bounds: Dict[str, Dict[str, float]] = {}
        for idx, name in enumerate(param_names):
            vals = np.asarray([t.params[name] for t in comp_trials], dtype=float)
            bounds[name] = {"min": float(vals.min()), "max": float(vals.max())}

        rows.append(
            {
                "component_id": comp_id,
                "size": int(len(comp_trials)),
                "objective_mean": float(objective_values.mean()),
                "objective_best": float(objective_values.max()),
                "objective_drift_within_component": float(objective_values.std(ddof=0)),
                "local_width_by_dim": local_width_by_dim,
                "active_dim_count": active_dim_count,
                "flat_volume_proxy": flat_volume_proxy,
                "bounds": bounds,
            }
        )

    if not rows:
        rows = [
            {
                "component_id": 0,
                "size": len(kept),
                "objective_mean": float(np.mean([t.goals.objective_score for t in kept])),
                "objective_best": float(np.max([t.goals.objective_score for t in kept])),
                "objective_drift_within_component": float(np.std([t.goals.objective_score for t in kept], ddof=0)),
                "local_width_by_dim": {name: 0.0 for name in param_names},
                "active_dim_count": len(param_names),
                "flat_volume_proxy": 0.0,
                "bounds": {
                    name: {"min": float(min(t.params[name] for t in kept)), "max": float(max(t.params[name] for t in kept))}
                    for name in param_names
                },
            }
        ]

    rows.sort(
        key=lambda r: (
            -float(r["objective_best"]),
            -float(r["flat_volume_proxy"]),
            int(r["active_dim_count"]),
        )
    )

    top_obj = float(rows[0]["objective_best"])
    tolerance_abs = abs(top_obj) * (performance_tolerance_pct / 100.0)
    comparable = [r for r in rows if float(r["objective_best"]) >= top_obj - tolerance_abs]
    comparable.sort(key=lambda r: (-float(r["flat_volume_proxy"]), int(r["active_dim_count"]), -int(r["size"])))
    selected = comparable[0]

    return {
        "selected_component": selected,
        "components": rows,
        "feasible_trial_count": len(feasible),
        "high_quality_trial_count": len(kept),
        "performance_tolerance_pct": performance_tolerance_pct,
    }


def write_region_summary(path: str, payload: Dict[str, object]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
