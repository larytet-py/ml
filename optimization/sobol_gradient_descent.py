from __future__ import annotations

import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .constrained_bo import ParamSpec, TrialResult
from .goal_registry import GoalEvaluation, evaluate_goals
from .goals import Goal

_SOBOL_BALANCE_WARNING = r"The balance properties of Sobol' points require n to be a power of 2\."

_PROCESS_EVALUATOR: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None


def _process_init(evaluator: Callable[[Dict[str, float]], Dict[str, float]]) -> None:
    global _PROCESS_EVALUATOR
    _PROCESS_EVALUATOR = evaluator


def _process_eval_params(params: Dict[str, float]) -> Dict[str, float]:
    if _PROCESS_EVALUATOR is None:
        raise RuntimeError("Process evaluator is not initialized.")
    return _PROCESS_EVALUATOR(params)


@dataclass
class SobolSeedSummary:
    seed_trial_id: int
    seed_params: Dict[str, float]
    best_local_trial_id: int
    best_local_objective: float
    best_local_penalized_score: float
    local_trial_count: int
    feasible_fraction: float
    stable_fraction: float
    stable_l2_radius_mean: float
    stable_linf_radius_max: float


class ConstrainedSobolGradientOptimizer:
    def __init__(
        self,
        search_space: Dict[str, ParamSpec],
        goals: List[Goal],
        evaluator: Callable[[Dict[str, float]], Dict[str, float]],
        trials_parquet: str,
        eval_cache_parquet: Optional[str] = None,
        seed: int = 42,
        sobol_samples: int = 256,
        top_ratio: float = 0.1,
        local_probe_per_seed: int = 10,
        local_probe_radius: float = 0.08,
        gradient_steps: int = 4,
        gradient_step_size: float = 0.03,
        gradient_learning_rate: float = 0.45,
        objective_tolerance_pct: float = 2.0,
        constraint_penalty: float = 1000.0,
        force_rebuild: bool = False,
        progress_interval_seconds: float = 5.0,
        checkpoint_interval_seconds: float = 60.0,
        max_eval_cache_entries: int = -1,
        parallel_backend: str = "thread",
        workers: int = 1,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        if sobol_samples <= 0:
            raise ValueError("sobol_samples must be > 0")
        if not (0 < top_ratio <= 1):
            raise ValueError("top_ratio must be in (0, 1]")
        if local_probe_per_seed < 0:
            raise ValueError("local_probe_per_seed must be >= 0")
        if gradient_steps < 0:
            raise ValueError("gradient_steps must be >= 0")
        if gradient_step_size <= 0:
            raise ValueError("gradient_step_size must be > 0")
        if gradient_learning_rate <= 0:
            raise ValueError("gradient_learning_rate must be > 0")

        self.search_space = search_space
        self.goals = goals
        self.evaluator = evaluator
        self.trials_parquet = Path(trials_parquet)
        self.eval_cache_parquet = Path(eval_cache_parquet) if eval_cache_parquet else None
        self.seed = int(seed)
        self.sobol_samples = int(sobol_samples)
        self.top_ratio = float(top_ratio)
        self.local_probe_per_seed = int(local_probe_per_seed)
        self.local_probe_radius = float(local_probe_radius)
        self.gradient_steps = int(gradient_steps)
        self.gradient_step_size = float(gradient_step_size)
        self.gradient_learning_rate = float(gradient_learning_rate)
        self.objective_tolerance_pct = float(objective_tolerance_pct)
        self.constraint_penalty = float(constraint_penalty)
        self.force_rebuild = bool(force_rebuild)
        self.progress_interval_seconds = float(progress_interval_seconds)
        self.checkpoint_interval_seconds = max(0.0, float(checkpoint_interval_seconds))
        self.max_eval_cache_entries = int(max_eval_cache_entries)
        self.eval_cache_enabled = self.max_eval_cache_entries != 0
        self.parallel_backend = str(parallel_backend).strip().lower()
        if self.parallel_backend not in {"thread", "process"}:
            raise ValueError("parallel_backend must be one of: thread, process")
        if self.parallel_backend == "process" and self.eval_cache_enabled:
            raise ValueError("Process backend requires eval cache disabled (set max_eval_cache_entries=0).")
        self.workers = max(1, int(workers))
        self.logger = logger or (lambda message: print(message, flush=True))

        self.param_names = list(search_space)
        self.dim = len(self.param_names)
        self.rng = np.random.default_rng(self.seed)

        self._trial_rows: List[Dict[str, object]] = []
        self._eval_cache: Dict[str, Dict[str, float]] = {}
        self._eval_cache_lock = Lock()
        self._cache_dirty = False
        self._executor: Optional[ThreadPoolExecutor | ProcessPoolExecutor] = None
        self._trials_parquet_write_count = 0
        self._last_persisted_trials_rows = 0
        self._eval_cache_parquet_write_count = 0
        self._last_persisted_eval_cache_rows = 0
        if self.eval_cache_enabled and not self.force_rebuild:
            self._load_eval_cache()

    def _log(self, message: str) -> None:
        self.logger(message)

    def _normalize(self, params: Dict[str, float]) -> np.ndarray:
        values = []
        for name in self.param_names:
            spec = self.search_space[name]
            width = float(spec.high - spec.low)
            if width <= 0:
                values.append(0.0)
            else:
                values.append((float(params[name]) - float(spec.low)) / width)
        return np.asarray(values, dtype=float)

    def _denormalize(self, x01: np.ndarray) -> Dict[str, float]:
        x = np.clip(np.asarray(x01, dtype=float), 0.0, 1.0)
        params: Dict[str, float] = {}
        for i, name in enumerate(self.param_names):
            spec = self.search_space[name]
            raw = float(spec.low) + float(x[i]) * float(spec.high - spec.low)
            if spec.is_int:
                params[name] = float(int(round(raw)))
            else:
                params[name] = float(raw)
        return params

    def _penalized(self, objective_score: float, violations: Dict[str, float]) -> float:
        total_violation = float(sum(max(0.0, v) for v in violations.values()))
        return objective_score - self.constraint_penalty * total_violation

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

    def _trials_header(self) -> List[str]:
        columns = [
            "trial_id",
            "phase",
            "parent_trial_id",
            "seed_rank",
            "penalized_score",
            "feasible",
            "objective_score",
        ]
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

    def _trial_to_row(
        self,
        trial: TrialResult,
        *,
        phase: str,
        parent_trial_id: Optional[int],
        seed_rank: Optional[int],
    ) -> Dict[str, object]:
        row: Dict[str, object] = {
            "trial_id": trial.trial_id,
            "phase": phase,
            "parent_trial_id": parent_trial_id,
            "seed_rank": seed_rank,
            "penalized_score": trial.penalized_score,
            "feasible": int(trial.goals.feasible),
            "objective_score": trial.goals.objective_score,
        }
        row.update({f"param__{k}": v for k, v in trial.params.items()})
        for key in {
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
        }:
            row[f"metric__{key}"] = trial.metrics.get(key)
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
        self._trials_parquet_write_count += 1
        self._last_persisted_trials_rows = int(len(df))

    def _persist_eval_cache(self) -> None:
        if self.eval_cache_parquet is None or not self.eval_cache_enabled:
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
        self._eval_cache_parquet_write_count += 1
        self._last_persisted_eval_cache_rows = int(len(rows))
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

    def _evaluate_trial(
        self,
        *,
        trial_id: int,
        params: Dict[str, float],
    ) -> tuple[TrialResult, bool]:
        if not self.eval_cache_enabled:
            metrics = self.evaluator(params)
            goals_eval = evaluate_goals(self.goals, metrics)
            trial = TrialResult(
                trial_id=trial_id,
                params=dict(params),
                metrics=metrics,
                goals=goals_eval,
                penalized_score=self._penalized(goals_eval.objective_score, goals_eval.violations),
            )
            return trial, False

        cache_key = self._params_cache_key(params)
        with self._eval_cache_lock:
            cached = self._eval_cache.get(cache_key)
        cache_hit = cached is not None
        if cache_hit:
            metrics = dict(cached)
        else:
            metrics = self.evaluator(params)
            with self._eval_cache_lock:
                existing = self._eval_cache.get(cache_key)
                if existing is None:
                    if self.max_eval_cache_entries < 0 or len(self._eval_cache) < self.max_eval_cache_entries:
                        self._eval_cache[cache_key] = dict(metrics)
                        self._cache_dirty = True
                else:
                    metrics = dict(existing)
                    cache_hit = True

        goals_eval = evaluate_goals(self.goals, metrics)
        trial = TrialResult(
            trial_id=trial_id,
            params=dict(params),
            metrics=metrics,
            goals=goals_eval,
            penalized_score=self._penalized(goals_eval.objective_score, goals_eval.violations),
        )
        return trial, cache_hit

    def _evaluate_batch(
        self,
        *,
        trial_id_start: int,
        params_batch: Sequence[Dict[str, float]],
        max_workers: Optional[int] = None,
    ) -> tuple[List[tuple[TrialResult, bool]], int]:
        params_list = list(params_batch)
        if not params_list:
            return [], trial_id_start
        worker_count = max(1, int(max_workers if max_workers is not None else self.workers))
        items = [(trial_id_start + i, params) for i, params in enumerate(params_list)]

        if worker_count <= 1 or len(items) <= 1:
            out = [self._evaluate_trial(trial_id=item[0], params=item[1]) for item in items]
            return out, trial_id_start + len(items)

        if self.parallel_backend == "process":
            if self._executor is None:
                self._executor = ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=_process_init,
                    initargs=(self.evaluator,),
                )
            metrics_batch = list(self._executor.map(_process_eval_params, params_list))
            out = []
            for i, metrics in enumerate(metrics_batch):
                trial_id = trial_id_start + i
                params = params_list[i]
                goals_eval = evaluate_goals(self.goals, metrics)
                trial = TrialResult(
                    trial_id=trial_id,
                    params=dict(params),
                    metrics=metrics,
                    goals=goals_eval,
                    penalized_score=self._penalized(goals_eval.objective_score, goals_eval.violations),
                )
                out.append((trial, False))
            return out, trial_id_start + len(items)

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=worker_count)
        out = list(
            self._executor.map(
                lambda item: self._evaluate_trial(trial_id=item[0], params=item[1]),
                items,
            )
        )
        return out, trial_id_start + len(items)

    def _sobol_points(self, n: int) -> np.ndarray:
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.seed)
        # random() lets us request arbitrary n, while still leveraging Sobol low-discrepancy structure.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_SOBOL_BALANCE_WARNING, category=UserWarning)
            return sampler.random(n=n)

    def _local_probe_points(self, center: np.ndarray, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, self.dim), dtype=float)
        noise = self.rng.normal(loc=0.0, scale=self.local_probe_radius, size=(n, self.dim))
        return np.clip(center[None, :] + noise, 0.0, 1.0)

    def _local_probe_points_for_seed(self, center: np.ndarray, n: int, seed_trial_id: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, self.dim), dtype=float)
        # Use per-seed RNG so restart/recompute paths are deterministic and independent of global call order.
        seed_value = (self.seed * 1_000_003 + int(seed_trial_id) * 97_531 + 17) % (2**63 - 1)
        local_rng = np.random.default_rng(seed_value)
        noise = local_rng.normal(loc=0.0, scale=self.local_probe_radius, size=(n, self.dim))
        return np.clip(center[None, :] + noise, 0.0, 1.0)

    def _trial_from_row(self, row: Dict[str, object]) -> TrialResult:
        trial_id = int(row["trial_id"])
        params: Dict[str, float] = {}
        for name in self.param_names:
            params[name] = float(row.get(f"param__{name}", 0.0))

        metrics: Dict[str, float] = {}
        for key in {
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
        }:
            value = row.get(f"metric__{key}", 0.0)
            metrics[key] = float(0.0 if pd.isna(value) else value)

        values: Dict[str, float] = {}
        violations: Dict[str, float] = {}
        for goal in self.goals:
            v = row.get(f"goal__{goal.name}", 0.0)
            d = row.get(f"viol__{goal.name}", 0.0)
            values[goal.name] = float(0.0 if pd.isna(v) else v)
            violations[goal.name] = float(0.0 if pd.isna(d) else d)

        feasible_raw = row.get("feasible", 0)
        feasible = bool(int(feasible_raw)) if not pd.isna(feasible_raw) else False
        objective_raw = row.get("objective_score", 0.0)
        objective_score = float(0.0 if pd.isna(objective_raw) else objective_raw)
        penalized_raw = row.get("penalized_score", 0.0)
        penalized = float(0.0 if pd.isna(penalized_raw) else penalized_raw)

        goals_eval = GoalEvaluation(
            values=values,
            violations=violations,
            feasible=feasible,
            objective_score=objective_score,
        )
        return TrialResult(
            trial_id=trial_id,
            params=params,
            metrics=metrics,
            goals=goals_eval,
            penalized_score=penalized,
        )

    def _load_trials_state(self) -> tuple[List[TrialResult], Dict[int, int]]:
        if not self.trials_parquet.exists():
            return [], {}
        df = pd.read_parquet(self.trials_parquet)
        if df.empty:
            return [], {}

        if "trial_id" not in df.columns:
            return [], {}
        df = df.sort_values("trial_id").reset_index(drop=True)
        rows = df.to_dict(orient="records")

        self._trial_rows = rows
        self._last_persisted_trials_rows = int(len(rows))
        self._trials_parquet_write_count = max(1, self._trials_parquet_write_count)

        all_trials: List[TrialResult] = []
        lineage: Dict[int, int] = {}
        for row in rows:
            trial = self._trial_from_row(row)
            all_trials.append(trial)
            parent_raw = row.get("parent_trial_id")
            if parent_raw is None or pd.isna(parent_raw):
                lineage[trial.trial_id] = trial.trial_id
            else:
                lineage[trial.trial_id] = int(parent_raw)

        return all_trials, lineage

    def _drop_trials_by_ids(self, trial_ids: set[int], all_trials: List[TrialResult], lineage: Dict[int, int]) -> None:
        if not trial_ids:
            return
        all_trials[:] = [t for t in all_trials if t.trial_id not in trial_ids]
        self._trial_rows = [r for r in self._trial_rows if int(r.get("trial_id", -1)) not in trial_ids]
        for tid in list(lineage.keys()):
            if tid in trial_ids:
                del lineage[tid]

    def _estimate_gradient(
        self,
        *,
        x_center: np.ndarray,
        trial_id_start: int,
        parent_trial_id: int,
        seed_rank: int,
    ) -> tuple[np.ndarray, List[tuple[TrialResult, str, Optional[int], Optional[int], bool]], int]:
        grad = np.zeros(self.dim, dtype=float)
        emitted: List[tuple[TrialResult, str, Optional[int], Optional[int], bool]] = []
        params_batch: List[Dict[str, float]] = []
        for j in range(self.dim):
            step_vec = np.zeros(self.dim, dtype=float)
            step_vec[j] = self.gradient_step_size
            x_plus = np.clip(x_center + step_vec, 0.0, 1.0)
            x_minus = np.clip(x_center - step_vec, 0.0, 1.0)
            params_batch.append(self._denormalize(x_plus))
            params_batch.append(self._denormalize(x_minus))

        evaluated, next_trial_id = self._evaluate_batch(trial_id_start=trial_id_start, params_batch=params_batch)
        for trial, cache_hit in evaluated:
            emitted.append((trial, "gradient_probe", parent_trial_id, seed_rank, cache_hit))

        for j in range(self.dim):
            trial_plus = evaluated[2 * j][0]
            trial_minus = evaluated[2 * j + 1][0]
            diff = float(trial_plus.penalized_score - trial_minus.penalized_score)
            grad[j] = diff / max(2.0 * self.gradient_step_size, 1e-12)

        return grad, emitted, next_trial_id

    def _build_seed_summaries(
        self,
        all_trials: Sequence[TrialResult],
        lineage: Dict[int, int],
        top_seeds: Sequence[TrialResult],
    ) -> List[SobolSeedSummary]:
        best_obj = max((t.goals.objective_score for t in all_trials if t.goals.feasible), default=None)
        tolerance_abs = 0.0
        if best_obj is not None:
            tolerance_abs = abs(best_obj) * (self.objective_tolerance_pct / 100.0)

        summaries: List[SobolSeedSummary] = []
        for seed_trial in top_seeds:
            related = [t for t in all_trials if lineage.get(t.trial_id, t.trial_id) == seed_trial.trial_id]
            if not related:
                continue
            best_local = max(related, key=lambda t: t.penalized_score)
            feasible_fraction = float(sum(1 for t in related if t.goals.feasible)) / float(len(related))

            if best_obj is None:
                stable_trials: List[TrialResult] = []
            else:
                stable_trials = [
                    t
                    for t in related
                    if t.goals.feasible and t.goals.objective_score >= float(best_obj) - tolerance_abs
                ]
            stable_fraction = float(len(stable_trials)) / float(len(related))

            seed_x = self._normalize(seed_trial.params)
            if stable_trials:
                stable_deltas = [self._normalize(t.params) - seed_x for t in stable_trials]
                l2_values = [float(np.linalg.norm(d)) for d in stable_deltas]
                linf_values = [float(np.max(np.abs(d))) for d in stable_deltas]
                stable_l2_radius_mean = float(np.mean(l2_values))
                stable_linf_radius_max = float(np.max(linf_values))
            else:
                stable_l2_radius_mean = 0.0
                stable_linf_radius_max = 0.0

            summaries.append(
                SobolSeedSummary(
                    seed_trial_id=seed_trial.trial_id,
                    seed_params=dict(seed_trial.params),
                    best_local_trial_id=best_local.trial_id,
                    best_local_objective=float(best_local.goals.objective_score),
                    best_local_penalized_score=float(best_local.penalized_score),
                    local_trial_count=len(related),
                    feasible_fraction=feasible_fraction,
                    stable_fraction=stable_fraction,
                    stable_l2_radius_mean=stable_l2_radius_mean,
                    stable_linf_radius_max=stable_linf_radius_max,
                )
            )

        summaries.sort(
            key=lambda s: (
                -float(s.stable_fraction),
                -float(s.stable_linf_radius_max),
                -float(s.best_local_penalized_score),
            )
        )
        return summaries

    def _checkpoint(self, *, force: bool = False) -> None:
        self._persist_trials()
        if self._cache_dirty:
            self._persist_eval_cache()
        self._log(
            "[sobol] parquet checkpoint "
            f"trials_parquet_rows={self._last_persisted_trials_rows} "
            f"trials_parquet_writes={self._trials_parquet_write_count} "
            f"eval_cache_rows={self._last_persisted_eval_cache_rows} "
            f"eval_cache_writes={self._eval_cache_parquet_write_count}"
        )
        if force:
            self._log("[sobol] checkpoint persisted (final)")

    def _maybe_checkpoint(self, *, now: float, last_checkpoint_time: float) -> float:
        if self.checkpoint_interval_seconds <= 0:
            return last_checkpoint_time
        if now - last_checkpoint_time < self.checkpoint_interval_seconds:
            return last_checkpoint_time
        self._checkpoint(force=False)
        return now

    def run(self) -> Dict[str, object]:
        if self.force_rebuild:
            self._eval_cache = {}
            self._cache_dirty = False
            self._trial_rows = []

        all_trials: List[TrialResult] = []
        lineage: Dict[int, int] = {}
        if not self.force_rebuild:
            all_trials, lineage = self._load_trials_state()

        trial_id = (max((t.trial_id for t in all_trials), default=-1) + 1)
        cache_hits = 0
        start_time = time.monotonic()
        last_progress_time = start_time
        last_checkpoint_time = start_time
        top_seeds: List[TrialResult] = []
        sobol_trials: List[TrialResult] = []
        best_penalized_trial: Optional[TrialResult] = None
        best_feasible_trial: Optional[TrialResult] = None
        best_itm_trial_id: Optional[int] = None
        best_itm_value: Optional[float] = None
        best_itm_drawdown: Optional[float] = None
        best_itm_avg_pnl: Optional[float] = None
        best_itm_total_trades: Optional[float] = None
        best_itm_rank: Optional[tuple[float, float, float]] = None

        def update_best_itm(trial: TrialResult) -> None:
            nonlocal best_itm_trial_id, best_itm_value, best_itm_drawdown, best_itm_avg_pnl, best_itm_total_trades, best_itm_rank
            if not trial.goals.feasible:
                return
            itm = float(trial.metrics.get("itm_expiries", 0.0))
            drawdown = float(trial.metrics.get("max_drawdown", 0.0))
            avg_pnl = float(trial.metrics.get("avg_pnl", 0.0))
            total_trades = float(trial.metrics.get("total", 0.0))
            rank = (itm, abs(drawdown), -avg_pnl)
            if best_itm_rank is None or rank < best_itm_rank:
                best_itm_rank = rank
                best_itm_trial_id = int(trial.trial_id)
                best_itm_value = itm
                best_itm_drawdown = drawdown
                best_itm_avg_pnl = avg_pnl
                best_itm_total_trades = total_trades

        def update_best_trials(trial: TrialResult) -> None:
            nonlocal best_penalized_trial, best_feasible_trial
            if best_penalized_trial is None or trial.penalized_score > best_penalized_trial.penalized_score:
                best_penalized_trial = trial
            if trial.goals.feasible and (
                best_feasible_trial is None or trial.goals.objective_score > best_feasible_trial.goals.objective_score
            ):
                best_feasible_trial = trial

        def _best_run_progress_part() -> str:
            if best_penalized_trial is None:
                return "best_trial_id=na best_penalized=na best_objective=na best_total=na best_total_pnl=na best_itm=na"
            t = best_penalized_trial
            return (
                f"best_trial_id={t.trial_id} "
                f"best_penalized={float(t.penalized_score):.6f} "
                f"best_objective={float(t.goals.objective_score):.6f} "
                f"best_total={float(t.metrics.get('total', 0.0)):.0f} "
                f"best_total_pnl={float(t.metrics.get('total_pnl', 0.0)):.6f} "
                f"best_itm={float(t.metrics.get('itm_expiries', 0.0)):.6f}"
            )

        def _region_progress_part() -> str:
            if not top_seeds:
                return "region_seed_id=na good_region_dims=0 flatness=na region_radius_linf=0.000 top_dim_spread01=na"

            related_by_seed: Dict[int, List[TrialResult]] = {int(seed.trial_id): [] for seed in top_seeds}
            for t in all_trials:
                root = int(lineage.get(t.trial_id, t.trial_id))
                if root in related_by_seed:
                    related_by_seed[root].append(t)

            best_obj = float(best_feasible_trial.goals.objective_score) if best_feasible_trial is not None else None
            tolerance_abs = abs(best_obj) * (self.objective_tolerance_pct / 100.0) if best_obj is not None else 0.0

            best_rank: Optional[tuple[float, float, float]] = None
            best_parts = "region_seed_id=na good_region_dims=0 flatness=na region_radius_linf=0.000 top_dim_spread01=na"
            for seed_trial in top_seeds:
                seed_id = int(seed_trial.trial_id)
                related = related_by_seed.get(seed_id, [])
                if not related:
                    continue

                if best_obj is None:
                    stable_trials: List[TrialResult] = []
                else:
                    stable_trials = [
                        t
                        for t in related
                        if t.goals.feasible and t.goals.objective_score >= float(best_obj) - tolerance_abs
                    ]
                stable_fraction = float(len(stable_trials)) / float(len(related))
                local_best_penalized = float(max(related, key=lambda t: t.penalized_score).penalized_score)

                if stable_trials:
                    seed_x = self._normalize(seed_trial.params)
                    stable_x = np.asarray([self._normalize(t.params) for t in stable_trials], dtype=float)
                    span = np.max(np.abs(stable_x - seed_x[None, :]), axis=0)
                    stable_norms = np.linalg.norm(stable_x - seed_x[None, :], axis=1)
                    region_radius_linf = float(np.max(span))
                    good_region_dims = int(np.sum(span >= self.gradient_step_size))
                    top_idx = np.argsort(span)[::-1][: min(3, self.dim)]
                    top_dim_spread = ",".join(
                        f"{self.param_names[int(i)]}:{float(span[int(i)]):.3f}"
                        for i in top_idx
                    )
                    flatness = float(stable_fraction)
                    region_l2_mean = float(np.mean(stable_norms))
                else:
                    region_radius_linf = 0.0
                    good_region_dims = 0
                    top_dim_spread = "na"
                    flatness = 0.0
                    region_l2_mean = 0.0

                rank = (flatness, region_radius_linf, local_best_penalized)
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_parts = (
                        f"region_seed_id={seed_id} "
                        f"good_region_dims={good_region_dims} "
                        f"flatness={flatness:.3f} "
                        f"region_radius_linf={region_radius_linf:.3f} "
                        f"region_radius_l2_mean={region_l2_mean:.3f} "
                        f"top_dim_spread01={top_dim_spread}"
                    )

            return best_parts

        def _maybe_log_phase2_progress(*, now: float, seed_rank: int, stage: str, force: bool = False) -> None:
            nonlocal last_progress_time
            if not force:
                if self.progress_interval_seconds <= 0:
                    return
                if now - last_progress_time < self.progress_interval_seconds:
                    return

            elapsed_seconds = now - start_time
            self._log(
                f"[sobol] phase2 progress seed_rank={seed_rank}/{len(top_seeds)} stage={stage} "
                f"{_best_run_progress_part()} {_region_progress_part()} "
                f"cache_hits={cache_hits} trials_total={len(all_trials)} elapsed_seconds={elapsed_seconds:.1f}"
            )
            last_progress_time = now

        for t in all_trials:
            update_best_itm(t)
            update_best_trials(t)

        try:
            phase_by_trial_id: Dict[int, str] = {}
            for row in self._trial_rows:
                tid = int(row.get("trial_id", -1))
                if tid < 0:
                    continue
                phase_by_trial_id[tid] = str(row.get("phase", ""))

            sobol_trials = [t for t in all_trials if phase_by_trial_id.get(t.trial_id) == "sobol"]
            sobol_trials.sort(key=lambda t: t.trial_id)
            sobol_completed = len(sobol_trials)
            sobol_points = self._sobol_points(self.sobol_samples)
            self._log(f"[sobol] sampling n={len(sobol_points)} dim={self.dim}")
            sobol_chunk_size = max(1, min(len(sobol_points), self.workers * 8))
            if sobol_completed > 0:
                self._log(
                    f"[sobol] resume detected existing_trials={len(all_trials)} "
                    f"existing_sobol_trials={sobol_completed}"
                )
            sobol_resume_start = sobol_completed
            if sobol_completed > len(sobol_points):
                # Resume behavior: if cache already has more Sobol trials than requested,
                # treat phase 1 as completed and continue to seed selection/local search.
                self._log(
                    f"[sobol] resume existing_sobol_trials={sobol_completed} exceeds "
                    f"requested_sobol_samples={len(sobol_points)}; treating phase1 as complete"
                )
                sobol_resume_start = len(sobol_points)

            for start in range(sobol_resume_start, len(sobol_points), sobol_chunk_size):
                chunk = sobol_points[start : start + sobol_chunk_size]
                params_batch = [self._denormalize(x) for x in chunk]
                evaluated, trial_id = self._evaluate_batch(trial_id_start=trial_id, params_batch=params_batch)
                for trial, cache_hit in evaluated:
                    all_trials.append(trial)
                    lineage[trial.trial_id] = trial.trial_id
                    self._trial_rows.append(self._trial_to_row(trial, phase="sobol", parent_trial_id=None, seed_rank=None))
                    update_best_itm(trial)
                    update_best_trials(trial)
                    if cache_hit:
                        cache_hits += 1

                now = time.monotonic()
                if self.progress_interval_seconds > 0 and now - last_progress_time >= self.progress_interval_seconds:
                    feasible_count = sum(1 for t in all_trials if t.goals.feasible)
                    best_penalized = max(t.penalized_score for t in all_trials)
                    elapsed_seconds = now - start_time
                    if best_itm_value is None:
                        best_itm_part = (
                            "best_itm=na best_itm_drawdown=na best_itm_avg_pnl=na "
                            "best_itm_total_trades=na best_itm_trial_id=na"
                        )
                    else:
                        best_itm_part = (
                            f"best_itm={best_itm_value:.6f} "
                            f"best_itm_drawdown={best_itm_drawdown:.6f} "
                            f"best_itm_avg_pnl={best_itm_avg_pnl:.6f} "
                            f"best_itm_total_trades={best_itm_total_trades:.0f} "
                            f"best_itm_trial_id={best_itm_trial_id}"
                        )
                    self._log(
                        f"[sobol] progress trial={trial_id}/{len(sobol_points)} feasible={feasible_count} "
                        f"cache_hits={cache_hits} best_penalized={best_penalized:.6f} "
                        f"{best_itm_part} "
                        f"elapsed_seconds={elapsed_seconds:.1f} "
                        f"trials_parquet_rows={self._last_persisted_trials_rows} "
                        f"trials_parquet_writes={self._trials_parquet_write_count}"
                    )
                    last_progress_time = now
                last_checkpoint_time = self._maybe_checkpoint(now=now, last_checkpoint_time=last_checkpoint_time)

            sobol_trial_ids = {
                int(row.get("trial_id", -1))
                for row in self._trial_rows
                if str(row.get("phase", "")) == "sobol"
            }
            sobol_trial_ids = {tid for tid in sobol_trial_ids if tid >= 0}
            sobol_trials = [t for t in all_trials if t.trial_id in sobol_trial_ids]
            sobol_trials.sort(key=lambda t: t.trial_id)
            sobol_trials = sobol_trials[: self.sobol_samples]

            # Seed selection criteria:
            # - At least 4 trades (metric__total > 3)
            # - Positive total PnL (metric__total_pnl > 0)
            # - Zero ITM expiries (metric__itm_expiries == 0)
            top_seeds = [
                t
                for t in sobol_trials
                if float(t.metrics.get("total", 0.0)) > 3.0
                and float(t.metrics.get("total_pnl", 0.0)) > 0.0
                and float(t.metrics.get("itm_expiries", 0.0)) == 0.0
            ]
            top_seeds.sort(key=lambda t: t.penalized_score, reverse=True)
            self._log(
                f"[sobol] finished samples={len(sobol_trials)} "
                f"selection=total_gt_3,total_pnl_gt_0,itm_expiries_eq_0 selected_seeds={len(top_seeds)}"
            )
            _maybe_log_phase2_progress(now=time.monotonic(), seed_rank=0, stage="seed_selection", force=True)

            seed_expected_trials = self.local_probe_per_seed + self.gradient_steps * (2 * self.dim + 1)
            for seed_trial in top_seeds:
                seed_id = seed_trial.trial_id
                child_ids = [
                    int(row.get("trial_id", -1))
                    for row in self._trial_rows
                    if (row.get("parent_trial_id") is not None and not pd.isna(row.get("parent_trial_id")))
                    and int(row.get("parent_trial_id")) == seed_id
                ]
                child_ids = [tid for tid in child_ids if tid >= 0]
                if 0 < len(child_ids) < seed_expected_trials:
                    self._log(
                        f"[sobol] resume partial seed seed_trial_id={seed_id} "
                        f"existing={len(child_ids)} expected={seed_expected_trials}; recomputing seed"
                    )
                    self._drop_trials_by_ids(set(child_ids), all_trials, lineage)
                    trial_id = max((t.trial_id for t in all_trials), default=-1) + 1

            for seed_rank, seed_trial in enumerate(top_seeds, start=1):
                seed_id = seed_trial.trial_id
                existing_child_count = sum(
                    1
                    for row in self._trial_rows
                    if (row.get("parent_trial_id") is not None and not pd.isna(row.get("parent_trial_id")))
                    and int(row.get("parent_trial_id")) == seed_id
                )
                if existing_child_count >= seed_expected_trials:
                    continue

                x_seed = self._normalize(seed_trial.params)
                probe_points = self._local_probe_points_for_seed(x_seed, self.local_probe_per_seed, seed_trial.trial_id)
                probe_params = [self._denormalize(x_probe) for x_probe in probe_points]
                probe_evals, trial_id = self._evaluate_batch(trial_id_start=trial_id, params_batch=probe_params)
                for trial, cache_hit in probe_evals:
                    all_trials.append(trial)
                    lineage[trial.trial_id] = seed_trial.trial_id
                    self._trial_rows.append(
                        self._trial_to_row(
                            trial,
                            phase="local_probe",
                            parent_trial_id=seed_trial.trial_id,
                            seed_rank=seed_rank,
                        )
                    )
                    update_best_itm(trial)
                    update_best_trials(trial)
                    if cache_hit:
                        cache_hits += 1
                _maybe_log_phase2_progress(now=time.monotonic(), seed_rank=seed_rank, stage="local_probe", force=False)
                last_checkpoint_time = self._maybe_checkpoint(
                    now=time.monotonic(),
                    last_checkpoint_time=last_checkpoint_time,
                )

                x_current = np.copy(x_seed)
                for _ in range(self.gradient_steps):
                    grad, emitted, trial_id = self._estimate_gradient(
                        x_center=x_current,
                        trial_id_start=trial_id,
                        parent_trial_id=seed_trial.trial_id,
                        seed_rank=seed_rank,
                    )
                    for trial, phase, parent, rank, cache_hit in emitted:
                        all_trials.append(trial)
                        lineage[trial.trial_id] = seed_trial.trial_id
                        self._trial_rows.append(self._trial_to_row(trial, phase=phase, parent_trial_id=parent, seed_rank=rank))
                        update_best_itm(trial)
                        update_best_trials(trial)
                        if cache_hit:
                            cache_hits += 1

                    grad_norm = float(np.linalg.norm(grad))
                    if grad_norm > 0:
                        x_current = np.clip(x_current + self.gradient_learning_rate * (grad / grad_norm), 0.0, 1.0)
                    trial_step, cache_hit = self._evaluate_trial(trial_id=trial_id, params=self._denormalize(x_current))
                    all_trials.append(trial_step)
                    lineage[trial_step.trial_id] = seed_trial.trial_id
                    self._trial_rows.append(
                        self._trial_to_row(
                            trial_step,
                            phase="gradient_step",
                            parent_trial_id=seed_trial.trial_id,
                            seed_rank=seed_rank,
                        )
                    )
                    update_best_itm(trial_step)
                    update_best_trials(trial_step)
                    if cache_hit:
                        cache_hits += 1
                    trial_id += 1
                    _maybe_log_phase2_progress(now=time.monotonic(), seed_rank=seed_rank, stage="gradient_step", force=False)
                    last_checkpoint_time = self._maybe_checkpoint(
                        now=time.monotonic(),
                        last_checkpoint_time=last_checkpoint_time,
                    )
        except KeyboardInterrupt:
            self._log("[sobol] interrupted by user; persisting latest checkpoint before exit")
            self._checkpoint(force=True)
            raise
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None

        self._checkpoint(force=True)

        feasible = [t for t in all_trials if t.goals.feasible]
        best_feasible = max(feasible, key=lambda t: t.goals.objective_score) if feasible else None
        best_penalized = max(all_trials, key=lambda t: t.penalized_score) if all_trials else None
        seed_summaries = self._build_seed_summaries(all_trials, lineage, top_seeds)

        elapsed = time.monotonic() - start_time
        self._log(
            f"[sobol] done total_trials={len(all_trials)} feasible={len(feasible)} cache_hits={cache_hits} elapsed={elapsed:.1f}s"
        )

        return {
            "trials": all_trials,
            "sobol_trials": sobol_trials,
            "top_seed_trial_ids": [t.trial_id for t in top_seeds],
            "seed_summaries": [s.__dict__ for s in seed_summaries],
            "best_feasible": best_feasible,
            "best_penalized": best_penalized,
            "cache_hits": cache_hits,
        }


def write_sobol_summary(path: str, payload: Dict[str, object]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
