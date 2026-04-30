#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from backtest_option_strategy_sobol_gradient import PrecomputedFeatureBacktester
from optimization.constrained_bo import ParamSpec
from optimization.gradient_descent import FiniteDifferenceGradientDescent
from optimization.sobol_sampling import SobolSampler

METRIC_KEYS: Tuple[str, ...] = (
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
)

DEFAULT_DIMENSIONS: Dict[str, Dict[str, Any]] = {
    "roc_window_size": {"min": 2, "max": 60, "type": "int"},
    "vol_window_size": {"min": 2, "max": 40, "type": "int"},
    "roc_threshold": {"min": -0.40, "max": 0.40, "type": "float"},
    "vol_threshold": {"min": 0.00, "max": 1.00, "type": "float"},
}

DIMENSION_ALIASES = {
    "roc_window": "roc_window_size",
    "vol_window": "vol_window_size",
}


@dataclass(frozen=True)
class DimensionSpec:
    name: str
    low: float
    high: float
    kind: str

    @property
    def is_int(self) -> bool:
        return self.kind == "int"


@dataclass(frozen=True)
class RunContext:
    symbol: str
    side: str
    start_date: Optional[str]
    end_date: Optional[str]


@dataclass(frozen=True)
class CandidateRun:
    params: Dict[str, float]
    phase: str
    parent_trial_id: Optional[int]
    seed_rank: Optional[int]


@dataclass
class Phase2CandidateStats:
    seeds_seen: int = 0
    seeds_completed: int = 0
    candidate_target: int = 0
    candidates_generated: int = 0
    candidates_reused: int = 0
    cache_or_duplicate_skips: int = 0
    candidate_shortfall: int = 0
    draws_attempted: int = 0


_WORKER_BACKTESTER: Optional[PrecomputedFeatureBacktester] = None
_WORKER_DIM_SPECS: Dict[str, DimensionSpec] = {}
_WORKER_BASE_KNOBS: Dict[str, Any] = {}
_WORKER_CONTEXT: Optional[RunContext] = None
_WORKER_EVAL_KWARGS: Dict[str, Any] = {}


def _compose_knobs_from_specs(
    base_knobs: Dict[str, Any],
    dim_specs: Dict[str, DimensionSpec],
    params: Dict[str, float],
) -> Dict[str, Any]:
    knobs = dict(base_knobs)
    for name, value in params.items():
        spec = dim_specs[name]
        if spec.is_int:
            knobs[name] = int(round(float(value)))
        else:
            knobs[name] = float(value)

    if knobs["roc_range_low"] > knobs["roc_range_high"]:
        knobs["roc_range_low"], knobs["roc_range_high"] = knobs["roc_range_high"], knobs["roc_range_low"]
    if knobs["vol_range_low"] > knobs["vol_range_high"]:
        knobs["vol_range_low"], knobs["vol_range_high"] = knobs["vol_range_high"], knobs["vol_range_low"]
    return knobs


def _install_process_worker_state(
    *,
    backtester: PrecomputedFeatureBacktester,
    dim_specs: Dict[str, DimensionSpec],
    base_knobs: Dict[str, Any],
    context: RunContext,
    eval_kwargs: Dict[str, Any],
) -> None:
    global _WORKER_BACKTESTER, _WORKER_DIM_SPECS, _WORKER_BASE_KNOBS, _WORKER_CONTEXT, _WORKER_EVAL_KWARGS
    _WORKER_BACKTESTER = backtester
    _WORKER_DIM_SPECS = dim_specs
    _WORKER_BASE_KNOBS = base_knobs
    _WORKER_CONTEXT = context
    _WORKER_EVAL_KWARGS = eval_kwargs


def _process_worker_initializer() -> None:
    if _WORKER_BACKTESTER is None or _WORKER_CONTEXT is None:
        raise RuntimeError("Multiprocessing worker state not initialized before fork.")


def _process_worker_backtest(params: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
    if _WORKER_BACKTESTER is None or _WORKER_CONTEXT is None:
        raise RuntimeError("Multiprocessing worker state is unavailable.")
    knobs = _compose_knobs_from_specs(_WORKER_BASE_KNOBS, _WORKER_DIM_SPECS, params)
    evaluation = _WORKER_BACKTESTER.evaluate(
        knobs_input=knobs,
        symbol=_WORKER_CONTEXT.symbol,
        start_date=_WORKER_CONTEXT.start_date,
        end_date=_WORKER_CONTEXT.end_date,
        risk_free_rate=float(_WORKER_EVAL_KWARGS["risk_free_rate"]),
        min_pricing_vol_annualized=float(_WORKER_EVAL_KWARGS["min_pricing_vol_annualized"]),
        contract_size=int(_WORKER_EVAL_KWARGS["contract_size"]),
    )
    metrics = {k: float(evaluation.metrics.get(k, 0.0)) for k in METRIC_KEYS}
    return bool(evaluation.feasible), metrics


class Orchestrator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.workers = self._resolve_workers(args.workers)

        self.backtester = PrecomputedFeatureBacktester.from_parquet(args.features_parquet)
        self.dim_specs = self._resolve_dimensions(args.window_config_yaml, self.backtester.df)
        self.search_space = {
            name: ParamSpec(low=spec.low, high=spec.high, is_int=spec.is_int)
            for name, spec in self.dim_specs.items()
        }
        self.sobol = SobolSampler(search_space=self.search_space, seed=int(args.seed))
        self.base_knobs = self._build_base_knobs(args)

        self.context = RunContext(
            symbol=str(args.symbol).upper(),
            side=str(args.side).lower(),
            start_date=args.start_date,
            end_date=args.end_date,
        )
        self.eval_kwargs = {
            "risk_free_rate": float(args.risk_free_rate),
            "min_pricing_vol_annualized": float(args.min_pricing_vol),
            "contract_size": int(args.contract_size),
        }
        _install_process_worker_state(
            backtester=self.backtester,
            dim_specs=self.dim_specs,
            base_knobs=self.base_knobs,
            context=self.context,
            eval_kwargs=self.eval_kwargs,
        )

        self.parquet_path = Path(args.trials_parquet)
        self.df = self._load_trials_df()
        self.next_trial_id = self._next_trial_id()
        self.key_to_trial_id: Dict[str, int] = {}
        self._rebuild_key_index()

        self.pending_rows: List[Dict[str, Any]] = []
        self.last_progress_at = 0.0
        self.last_checkpoint_at = time.monotonic()

        self.best_row: Optional[Dict[str, Any]] = self._compute_best_from_df(self.df)

    @staticmethod
    def _resolve_workers(requested_workers: int) -> int:
        cpu = os.cpu_count() or 1
        if requested_workers == 0:
            return max(1, cpu)
        if requested_workers > 0:
            return requested_workers
        return max(1, cpu + requested_workers)

    @staticmethod
    def _load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
        if not path:
            return {}
        p = Path(path)
        if not p.exists():
            return {}
        loaded = yaml.safe_load(p.read_text()) or {}
        if not isinstance(loaded, dict):
            return {}
        return loaded

    @staticmethod
    def _canonical_dim_name(name: str) -> str:
        return DIMENSION_ALIASES.get(name, name)

    @staticmethod
    def _derive_bounds_from_feature(df: pd.DataFrame, feature_name: str) -> Optional[Tuple[float, float]]:
        if feature_name not in df.columns:
            return None
        values = pd.to_numeric(df[feature_name], errors="coerce").to_numpy(dtype=float, copy=False)
        if values.size == 0:
            return None
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        return float(np.nanmin(finite)), float(np.nanmax(finite))

    @staticmethod
    def _coerce_kind(raw_kind: Optional[str], low: Optional[float], high: Optional[float]) -> str:
        if raw_kind:
            kind = str(raw_kind).strip().lower()
            if kind not in {"int", "float"}:
                raise ValueError(f"Unsupported dimension type '{raw_kind}'. Expected int|float.")
            return kind
        if low is not None and high is not None and float(low).is_integer() and float(high).is_integer():
            return "int"
        return "float"

    def _resolve_dimensions(self, config_path: Optional[str], features_df: pd.DataFrame) -> Dict[str, DimensionSpec]:
        config = self._load_yaml_config(config_path)
        optimization_cfg = config.get("optimization", {}) if isinstance(config, dict) else {}
        dimensions_cfg = optimization_cfg.get("dimensions", {})
        dims_in = dimensions_cfg if isinstance(dimensions_cfg, dict) else {}

        merged: Dict[str, Dict[str, Any]] = {}
        for name, spec in DEFAULT_DIMENSIONS.items():
            merged[name] = dict(spec)

        for raw_name, raw_spec in dims_in.items():
            if not isinstance(raw_spec, dict):
                continue
            merged[self._canonical_dim_name(str(raw_name))] = dict(raw_spec)

        resolved: Dict[str, DimensionSpec] = {}
        for raw_name, spec in merged.items():
            name = self._canonical_dim_name(raw_name)
            low_raw = spec.get("min")
            high_raw = spec.get("max")
            kind = self._coerce_kind(spec.get("type"), low_raw, high_raw)

            if low_raw is None or high_raw is None:
                derived = self._derive_bounds_from_feature(features_df, name)
                if derived is None:
                    raise ValueError(
                        f"Unable to resolve bounds for '{name}'. Provide min/max in YAML or feature parquet."
                    )
                low_raw, high_raw = derived

            low = float(low_raw)
            high = float(high_raw)
            if low > high:
                raise ValueError(f"Invalid bounds for '{name}': {low} > {high}")
            resolved[name] = DimensionSpec(name=name, low=low, high=high, kind=kind)

        if not resolved:
            raise ValueError("No optimization dimensions resolved.")
        return resolved

    @staticmethod
    def _build_base_knobs(args: argparse.Namespace) -> Dict[str, Any]:
        return {
            "side": args.side,
            "roc_window_size": int(args.roc_window_default),
            "roc_comparator": args.roc_comparator,
            "roc_threshold": float(args.roc_threshold_default),
            "roc_range_enabled": int(args.roc_range_enabled),
            "roc_range_low": float(args.roc_range_low),
            "roc_range_high": float(args.roc_range_high),
            "vol_window_size": int(args.vol_window_default),
            "vol_comparator": args.vol_comparator,
            "vol_threshold": float(args.vol_threshold_default),
            "vol_range_enabled": int(args.vol_range_enabled),
            "vol_range_low": float(args.vol_range_low),
            "vol_range_high": float(args.vol_range_high),
        }

    def _compose_knobs(self, params: Dict[str, float]) -> Dict[str, Any]:
        return _compose_knobs_from_specs(self.base_knobs, self.dim_specs, params)

    def _cache_key(self, params: Dict[str, float]) -> str:
        payload: Dict[str, Any] = {
            "symbol": self.context.symbol,
            "side": self.context.side,
            "start_date": self.context.start_date,
            "end_date": self.context.end_date,
        }
        for name in sorted(self.dim_specs):
            spec = self.dim_specs[name]
            value = float(params[name])
            payload[name] = int(round(value)) if spec.is_int else round(value, 12)
        return json.dumps(payload, sort_keys=True)

    def _normalize_params(self, params: Dict[str, float]) -> np.ndarray:
        out: List[float] = []
        for name, spec in self.dim_specs.items():
            width = float(spec.high - spec.low)
            if width <= 0:
                out.append(0.0)
                continue
            out.append((float(params[name]) - float(spec.low)) / width)
        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)

    def _denormalize_params(self, x01: Sequence[float]) -> Dict[str, float]:
        x = np.clip(np.asarray(x01, dtype=float), 0.0, 1.0)
        out: Dict[str, float] = {}
        for i, (name, spec) in enumerate(self.dim_specs.items()):
            raw = float(spec.low) + float(x[i]) * float(spec.high - spec.low)
            out[name] = float(int(round(raw))) if spec.is_int else float(raw)
        return out

    def _load_trials_df(self) -> pd.DataFrame:
        if not self.parquet_path.exists():
            return pd.DataFrame(columns=self._trial_columns())
        loaded = pd.read_parquet(self.parquet_path)
        for col in self._trial_columns():
            if col not in loaded.columns:
                loaded[col] = np.nan
        return loaded

    def _trial_columns(self) -> List[str]:
        cols = [
            "trial_id",
            "cache_key",
            "symbol",
            "side",
            "start_date",
            "end_date",
            "phase",
            "parent_trial_id",
            "seed_rank",
            "feasible",
            "objective_score",
        ]
        cols.extend(f"param__{name}" for name in self.dim_specs)
        cols.extend(f"metric__{name}" for name in METRIC_KEYS)
        return cols

    def _next_trial_id(self) -> int:
        if self.df.empty or "trial_id" not in self.df.columns:
            return 0
        values = pd.to_numeric(self.df["trial_id"], errors="coerce")
        if values.notna().sum() == 0:
            return 0
        return int(values.max()) + 1

    def _row_params(self, row: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, spec in self.dim_specs.items():
            value = float(row.get(f"param__{name}", spec.low))
            if spec.is_int:
                value = float(int(round(value)))
            out[name] = value
        return out

    def _rebuild_key_index(self) -> None:
        self.key_to_trial_id.clear()
        if self.df.empty:
            return
        records = self.df.to_dict(orient="records")
        for row in records:
            trial_id = int(row.get("trial_id", -1))
            cache_key = row.get("cache_key")
            if isinstance(cache_key, str) and cache_key:
                self.key_to_trial_id[cache_key] = trial_id
                continue
            params = self._row_params(row)
            computed = self._cache_key(params)
            self.key_to_trial_id[computed] = trial_id
            row["cache_key"] = computed
        self.df = pd.DataFrame(records)

    def _find_row_by_trial_id(self, trial_id: int) -> Optional[Dict[str, Any]]:
        if self.df.empty:
            return None
        subset = self.df[pd.to_numeric(self.df["trial_id"], errors="coerce") == int(trial_id)]
        if subset.empty:
            return None
        return subset.iloc[0].to_dict()

    @staticmethod
    def _rank_tuple(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
        total = float(row.get("metric__total", 0.0))
        itm = float(row.get("metric__itm_expiries", 0.0))
        drawdown = float(row.get("metric__max_drawdown", -1e12))
        total_pnl = float(row.get("metric__total_pnl", -1e12))
        return total_pnl, -itm, drawdown, total

    @staticmethod
    def _objective_from_metrics(metrics: Dict[str, float]) -> float:
        total = float(metrics.get("total", 0.0))
        itm = float(metrics.get("itm_expiries", 0.0))
        drawdown = float(metrics.get("max_drawdown", 0.0))
        total_pnl = float(metrics.get("total_pnl", 0.0))
        return (1_000_000.0 * total_pnl) - (10_000.0 * itm) + (100.0 * drawdown) + total

    def _compute_best_from_df(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if df.empty:
            return None
        best: Optional[Dict[str, Any]] = None
        for row in df.to_dict(orient="records"):
            if best is None or self._rank_tuple(row) > self._rank_tuple(best):
                best = row
        return best

    def _update_best(self, row: Dict[str, Any]) -> None:
        if self.best_row is None or self._rank_tuple(row) > self._rank_tuple(self.best_row):
            self.best_row = dict(row)

    def _log_progress(self, *, force: bool = False, prefix: str = "progress") -> None:
        if self.args.progress_seconds <= 0:
            return
        now = time.monotonic()
        if (not force) and (now - self.last_progress_at < self.args.progress_seconds):
            return
        self.last_progress_at = now
        if self.best_row is None:
            print(f"[{prefix}] trials={len(self.df)} best=none", flush=True)
            return
        print(
            f"[{prefix}] trials={len(self.df)} "
            f"best_trial_id={int(self.best_row['trial_id'])} "
            f"avg_pnl={float(self.best_row.get('metric__avg_pnl', 0.0)):.6f} "
            f"total_pnl={float(self.best_row.get('metric__total_pnl', 0.0)):.6f} "
            f"trades={float(self.best_row.get('metric__total', 0.0)):.0f} "
            f"itm={float(self.best_row.get('metric__itm_expiries', 0.0)):.6f}",
            flush=True,
        )

    def _persist_trials(self) -> None:
        if not self.pending_rows:
            return
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        append_df = pd.DataFrame(self.pending_rows)
        all_df = pd.concat([self.df, append_df], ignore_index=True)
        for col in self._trial_columns():
            if col not in all_df.columns:
                all_df[col] = np.nan
        all_df = all_df[self._trial_columns()]
        all_df.to_parquet(self.parquet_path, index=False)
        self.df = all_df
        self.pending_rows = []

    def _maybe_checkpoint(self, *, force: bool = False) -> None:
        if not self.pending_rows:
            return
        if self.args.checkpoint_seconds <= 0:
            if force:
                self._persist_trials()
            return
        now = time.monotonic()
        if force or (now - self.last_checkpoint_at >= self.args.checkpoint_seconds):
            self._persist_trials()
            self.last_checkpoint_at = now

    def _worker_backtest(self, params: Dict[str, float]) -> Tuple[bool, Dict[str, float], Dict[str, float]]:
        knobs = self._compose_knobs(params)
        evaluation = self.backtester.evaluate(
            knobs_input=knobs,
            symbol=self.context.symbol,
            start_date=self.context.start_date,
            end_date=self.context.end_date,
            risk_free_rate=float(self.eval_kwargs["risk_free_rate"]),
            min_pricing_vol_annualized=float(self.eval_kwargs["min_pricing_vol_annualized"]),
            contract_size=int(self.eval_kwargs["contract_size"]),
        )
        metrics = {k: float(evaluation.metrics.get(k, 0.0)) for k in METRIC_KEYS}
        objective = self._objective_from_metrics(metrics)
        return bool(evaluation.feasible), metrics, {"objective_score": float(objective)}

    def _row_from_worker_result(
        self,
        *,
        params: Dict[str, float],
        phase: str,
        parent_trial_id: Optional[int],
        seed_rank: Optional[int],
        feasible: bool,
        metrics: Dict[str, float],
        objective_score: float,
    ) -> Dict[str, Any]:
        cache_key = self._cache_key(params)
        row: Dict[str, Any] = {
            "trial_id": int(self.next_trial_id),
            "cache_key": cache_key,
            "symbol": self.context.symbol,
            "side": self.context.side,
            "start_date": self.context.start_date,
            "end_date": self.context.end_date,
            "phase": phase,
            "parent_trial_id": float(parent_trial_id) if parent_trial_id is not None else np.nan,
            "seed_rank": float(seed_rank) if seed_rank is not None else np.nan,
            "feasible": int(feasible),
            "objective_score": float(objective_score),
        }
        for name in self.dim_specs:
            row[f"param__{name}"] = float(params[name])
        for k in METRIC_KEYS:
            row[f"metric__{k}"] = float(metrics.get(k, 0.0))
        self.next_trial_id += 1
        self.key_to_trial_id[cache_key] = int(row["trial_id"])
        return row

    def _submit_candidates(self, candidates: Iterable[CandidateRun], *, phase_label: str) -> Tuple[int, int]:
        submitted = 0
        cache_hits = 0
        in_flight_keys: set[str] = set()
        futures: Dict[Future[Any], Tuple[Dict[str, float], str, Optional[int], Optional[int], str]] = {}

        with ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=mp.get_context("fork"),
            initializer=_process_worker_initializer,
        ) as executor:
            for cand in candidates:
                params = dict(cand.params)
                key = self._cache_key(params)
                if key in self.key_to_trial_id or key in in_flight_keys:
                    cache_hits += 1
                    continue

                future = executor.submit(_process_worker_backtest, params)
                futures[future] = (params, cand.phase, cand.parent_trial_id, cand.seed_rank, key)
                in_flight_keys.add(key)
                submitted += 1

                if len(futures) >= self.workers * 2:
                    self._drain_some_futures(futures, in_flight_keys=in_flight_keys)

                self._maybe_checkpoint(force=False)
                self._log_progress(prefix=phase_label)

            while futures:
                self._drain_some_futures(futures, in_flight_keys=in_flight_keys)
                self._maybe_checkpoint(force=False)
                self._log_progress(prefix=phase_label)

        self._maybe_checkpoint(force=True)
        self._log_progress(force=True, prefix=phase_label)
        return submitted, cache_hits

    def _drain_some_futures(
        self,
        futures: Dict[Future[Any], Tuple[Dict[str, float], str, Optional[int], Optional[int], str]],
        *,
        in_flight_keys: Optional[set[str]] = None,
    ) -> None:
        done, _ = wait(list(futures.keys()), timeout=0.2, return_when=FIRST_COMPLETED)
        if not done:
            return
        for future in done:
            params, phase, parent_trial_id, seed_rank, cache_key = futures.pop(future)
            try:
                feasible, metrics = future.result()
            finally:
                if in_flight_keys is not None:
                    in_flight_keys.discard(cache_key)
            row = self._row_from_worker_result(
                params=params,
                phase=phase,
                parent_trial_id=parent_trial_id,
                seed_rank=seed_rank,
                feasible=feasible,
                metrics=metrics,
                objective_score=float(self._objective_from_metrics(metrics)),
            )
            self.pending_rows.append(row)
            self._update_best(row)

    def _run_phase1(self) -> None:
        print(
            f"[phase1] start sobol_samples={self.args.sobol_samples} seed={self.args.seed} workers={self.workers}",
            flush=True,
        )
        phase_points = self.sobol.sample_phase1(int(self.args.sobol_samples)).params
        candidates = [
            CandidateRun(params=point, phase="phase1", parent_trial_id=None, seed_rank=None)
            for point in phase_points
        ]
        submitted, cache_hits = self._submit_candidates(candidates, phase_label="phase1")
        print(
            f"[phase1] complete total={len(candidates)} submitted={submitted} cache_hits={cache_hits}",
            flush=True,
        )

    def _params_matrix_normalized(self) -> np.ndarray:
        if self.df.empty:
            return np.zeros((0, len(self.dim_specs)), dtype=float)
        cols = [f"param__{name}" for name in self.dim_specs]
        data = self.df[cols].astype(float).to_numpy(dtype=float)
        out = np.zeros_like(data, dtype=float)
        for i, spec in enumerate(self.dim_specs.values()):
            width = float(spec.high - spec.low)
            if width <= 0:
                out[:, i] = 0.0
            else:
                out[:, i] = (data[:, i] - float(spec.low)) / width
        return np.clip(out, 0.0, 1.0)

    def _sorted_seed_rows(self) -> List[Dict[str, Any]]:
        if self.df.empty:
            return []
        allowed_phases = self._allowed_seed_phases()
        eligible: List[Dict[str, Any]] = []
        for row in self.df.to_dict(orient="records"):
            if allowed_phases is not None and str(row.get("phase", "")) not in allowed_phases:
                continue
            total = float(row.get("metric__total", 0.0))
            itm = float(row.get("metric__itm_expiries", 0.0))
            total_pnl = float(row.get("metric__total_pnl", 0.0))
            if (
                total > float(self.args.min_seed_trades)
                and itm < float(self.args.max_seed_itm)
                and total_pnl > 0.0
            ):
                eligible.append(row)
        eligible.sort(key=lambda r: self._rank_tuple(r), reverse=True)
        if not eligible:
            return []
        take = max(1, int(math.ceil(len(eligible) * float(self.args.seed_top_ratio))))
        return eligible[:take]

    def _allowed_seed_phases(self) -> Optional[set[str]]:
        raw = str(self.args.seed_phases).strip().lower()
        if raw in {"", "all", "*"}:
            return None
        return {part.strip() for part in raw.split(",") if part.strip()}

    def _iter_phase2_local_candidates(
        self,
        seed_rows: List[Dict[str, Any]],
        stats: Phase2CandidateStats,
    ) -> Iterator[CandidateRun]:
        if not seed_rows:
            return

        radius = float(self.args.local_radius)
        per_seed = int(self.args.local_probe_per_seed)
        if per_seed <= 0:
            return

        for idx, seed_row in enumerate(seed_rows, start=1):
            stats.seeds_seen = idx
            seed_params = self._row_params(seed_row)
            seed_tid = int(seed_row["trial_id"])
            center = self._normalize_params(seed_params)

            stats.candidate_target += per_seed
            generated_for_seed = 0
            generated_keys_for_seed: set[str] = set()
            attempts = 0
            draws_for_seed = 0
            max_draws = max(per_seed * 50, per_seed + 1000)
            while generated_for_seed < per_seed and draws_for_seed < max_draws:
                needed = per_seed - generated_for_seed
                batch_size = min(max(per_seed, needed), max_draws - draws_for_seed)
                if batch_size <= 0:
                    break
                sampler_seed = int(self.args.seed) + (idx * 1009) + (attempts * 104729)
                local_sampler = SobolSampler(search_space=self.search_space, seed=sampler_seed)
                local_unit = local_sampler.sample_unit(batch_size)
                attempts += 1
                draws_for_seed += batch_size
                stats.draws_attempted += batch_size
                for point in local_unit:
                    if generated_for_seed >= per_seed:
                        break
                    shifted = np.clip(center + ((point * 2.0) - 1.0) * radius, 0.0, 1.0)
                    params = self._denormalize_params(shifted)
                    key = self._cache_key(params)
                    if key in generated_keys_for_seed:
                        stats.cache_or_duplicate_skips += 1
                        continue
                    generated_keys_for_seed.add(key)
                    if key in self.key_to_trial_id:
                        generated_for_seed += 1
                        stats.candidates_reused += 1
                        continue
                    generated_for_seed += 1
                    stats.candidates_generated += 1
                    yield CandidateRun(
                        params=params,
                        phase="phase2",
                        parent_trial_id=seed_tid,
                        seed_rank=idx,
                    )
            if generated_for_seed >= per_seed:
                stats.seeds_completed += 1
            else:
                stats.candidate_shortfall += per_seed - generated_for_seed

    def _run_phase2(self) -> List[Dict[str, Any]]:
        seed_rows = self._sorted_seed_rows()
        if not seed_rows:
            print("[phase2] skipped no eligible seeds", flush=True)
            return []
        print(
            f"[phase2] eligible_seeds={len(seed_rows)} top_ratio={self.args.seed_top_ratio:.3f} "
            f"seed_phases={self.args.seed_phases} "
            f"min_seed_trades>{self.args.min_seed_trades} max_seed_itm<{self.args.max_seed_itm}",
            flush=True,
        )
        max_candidates = len(seed_rows) * int(self.args.local_probe_per_seed)
        print(
            f"[phase2] streaming_local_candidates seeds={len(seed_rows)} "
            f"per_seed={int(self.args.local_probe_per_seed)} "
            f"max_candidates={max_candidates} radius={float(self.args.local_radius):.6f}",
            flush=True,
        )
        stats = Phase2CandidateStats()
        candidates = self._iter_phase2_local_candidates(seed_rows, stats)
        submitted, cache_hits = self._submit_candidates(candidates, phase_label="phase2")
        print(
            f"[phase2] local_candidates={stats.candidates_generated} "
            f"reused_candidates={stats.candidates_reused} "
            f"target_new_candidates={stats.candidate_target} "
            f"seeds_scanned={stats.seeds_seen} "
            f"seeds_completed={stats.seeds_completed} "
            f"cache_or_duplicate_skips={stats.cache_or_duplicate_skips} "
            f"draws_attempted={stats.draws_attempted} "
            f"shortfall={stats.candidate_shortfall}",
            flush=True,
        )
        print(
            f"[phase2] complete submitted={submitted} cache_hits={cache_hits}",
            flush=True,
        )
        return seed_rows

    def _evaluate_sync_cached(
        self,
        *,
        params: Dict[str, float],
        phase: str,
        parent_trial_id: Optional[int],
        seed_rank: Optional[int],
    ) -> Dict[str, Any]:
        key = self._cache_key(params)
        trial_id = self.key_to_trial_id.get(key)
        if trial_id is not None:
            row = self._find_row_by_trial_id(trial_id)
            if row is not None:
                return row

        feasible, metrics, extra = self._worker_backtest(params)
        row = self._row_from_worker_result(
            params=params,
            phase=phase,
            parent_trial_id=parent_trial_id,
            seed_rank=seed_rank,
            feasible=feasible,
            metrics=metrics,
            objective_score=float(extra["objective_score"]),
        )
        self.pending_rows.append(row)
        self._update_best(row)
        self._maybe_checkpoint(force=False)
        self._log_progress(prefix="gradient")
        return row

    def _evaluate_gradient_batch_cached(
        self,
        *,
        requests: Sequence[Tuple[Dict[str, float], int, int]],
        executor: ProcessPoolExecutor,
    ) -> List[Dict[str, Any]]:
        rows_by_key: Dict[str, Dict[str, Any]] = {}
        futures: Dict[Future[Any], Tuple[Dict[str, float], int, int, str]] = {}
        ordered_keys: List[str] = []
        cache_hits = 0

        for params_in, parent_trial_id, seed_rank in requests:
            params = dict(params_in)
            key = self._cache_key(params)
            ordered_keys.append(key)
            if key in rows_by_key:
                cache_hits += 1
                continue
            trial_id = self.key_to_trial_id.get(key)
            if trial_id is not None:
                row = self._find_row_by_trial_id(trial_id)
                if row is not None:
                    rows_by_key[key] = row
                    cache_hits += 1
                    continue
            future = executor.submit(_process_worker_backtest, params)
            futures[future] = (params, parent_trial_id, seed_rank, key)

        while futures:
            done, _ = wait(list(futures.keys()), timeout=0.2, return_when=FIRST_COMPLETED)
            if not done:
                self._maybe_checkpoint(force=False)
                self._log_progress(prefix="gradient")
                continue
            for future in done:
                params, parent_trial_id, seed_rank, key = futures.pop(future)
                feasible, metrics = future.result()
                row = self._row_from_worker_result(
                    params=params,
                    phase="gradient",
                    parent_trial_id=parent_trial_id,
                    seed_rank=seed_rank,
                    feasible=feasible,
                    metrics=metrics,
                    objective_score=float(self._objective_from_metrics(metrics)),
                )
                self.pending_rows.append(row)
                self._update_best(row)
                rows_by_key[key] = row
            self._maybe_checkpoint(force=False)
            self._log_progress(prefix="gradient")

        self._maybe_checkpoint(force=False)
        self._log_progress(prefix="gradient")
        if cache_hits:
            # Keep the accounting visible without turning every cached probe into a progress line.
            pass
        return [rows_by_key[key] for key in ordered_keys]

    def _row_objective(self, row: Dict[str, Any]) -> float:
        metrics = {k: float(row.get(f"metric__{k}", 0.0)) for k in METRIC_KEYS}
        return self._objective_from_metrics(metrics)

    def _run_gradient(self, seed_rows: List[Dict[str, Any]]) -> None:
        if int(self.args.gradient_steps) <= 0 or not seed_rows:
            print("[gradient] skipped", flush=True)
            return

        print(
            f"[gradient] start seeds={len(seed_rows)} steps={self.args.gradient_steps} "
            f"step_size={self.args.gradient_step_size} lr={self.args.gradient_learning_rate}",
            flush=True,
        )

        steps = int(self.args.gradient_steps)
        step_size = float(self.args.gradient_step_size)
        learning_rate = float(self.args.gradient_learning_rate)
        dim = len(self.dim_specs)

        with ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=mp.get_context("fork"),
            initializer=_process_worker_initializer,
        ) as executor:
            for seed_idx, seed_row in enumerate(seed_rows, start=1):
                seed_tid = int(seed_row["trial_id"])
                x = self._normalize_params(self._row_params(seed_row))

                for step_index in range(steps + 1):
                    requests: List[Tuple[Dict[str, float], int, int]] = [
                        (self._denormalize_params(x), seed_tid, seed_idx)
                    ]
                    probe_pairs: List[Tuple[int, int, int]] = []

                    if step_index < steps:
                        for dim_index in range(dim):
                            offset = np.zeros(dim, dtype=float)
                            offset[dim_index] = step_size
                            plus_idx = len(requests)
                            requests.append((self._denormalize_params(np.clip(x + offset, 0.0, 1.0)), seed_tid, seed_idx))
                            minus_idx = len(requests)
                            requests.append((self._denormalize_params(np.clip(x - offset, 0.0, 1.0)), seed_tid, seed_idx))
                            probe_pairs.append((dim_index, plus_idx, minus_idx))

                    rows = self._evaluate_gradient_batch_cached(requests=requests, executor=executor)

                    if step_index == steps:
                        break

                    gradient = np.zeros(dim, dtype=float)
                    denominator = max(2.0 * step_size, 1e-12)
                    for dim_index, plus_idx, minus_idx in probe_pairs:
                        y_plus = self._row_objective(rows[plus_idx])
                        y_minus = self._row_objective(rows[minus_idx])
                        gradient[dim_index] = (y_plus - y_minus) / denominator

                    gradient_norm = float(np.linalg.norm(gradient))
                    if gradient_norm <= 0:
                        break

                    x = np.clip(x + learning_rate * (gradient / gradient_norm), 0.0, 1.0)

        self._maybe_checkpoint(force=True)
        self._log_progress(force=True, prefix="gradient")
        print("[gradient] complete", flush=True)

    def _rows_in_seed_areas(self, seed_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.df.empty or not seed_rows:
            return []

        matrix = self._params_matrix_normalized()
        if matrix.shape[0] == 0:
            return []

        radius = float(self.args.local_radius)
        keep = np.zeros(matrix.shape[0], dtype=bool)
        for seed_row in seed_rows:
            seed_params = self._row_params(seed_row)
            center = self._normalize_params(seed_params)
            distances = np.max(np.abs(matrix - center[None, :]), axis=1)
            keep |= distances <= radius

        kept_df = self.df[keep]
        return kept_df.to_dict(orient="records")

    def _write_final_json(self, seed_rows: List[Dict[str, Any]]) -> None:
        area_rows = self._rows_in_seed_areas(seed_rows)
        area_rows.sort(key=lambda row: self._rank_tuple(row), reverse=True)
        top_n = int(self.args.final_top_n)
        if top_n > 0:
            area_rows = area_rows[:top_n]

        out_rows: List[Dict[str, Any]] = []
        for row in area_rows:
            params = {name: float(row.get(f"param__{name}", 0.0)) for name in self.dim_specs}
            metrics = {
                "total": float(row.get("metric__total", 0.0)),
                "itm_expiries": float(row.get("metric__itm_expiries", 0.0)),
                "max_drawdown": float(row.get("metric__max_drawdown", 0.0)),
                "avg_pnl": float(row.get("metric__avg_pnl", 0.0)),
                "total_pnl": float(row.get("metric__total_pnl", 0.0)),
            }
            out_rows.append(
                {
                    "trial_id": int(row.get("trial_id", -1)),
                    "phase": row.get("phase"),
                    "parent_trial_id": None
                    if pd.isna(row.get("parent_trial_id"))
                    else int(float(row.get("parent_trial_id"))),
                    "seed_rank": None if pd.isna(row.get("seed_rank")) else int(float(row.get("seed_rank"))),
                    "params": params,
                    "metrics": metrics,
                }
            )

        payload = {
            "symbol": self.context.symbol,
            "side": self.context.side,
            "start_date": self.context.start_date,
            "end_date": self.context.end_date,
            "top_n": top_n,
            "seed_area_radius": float(self.args.local_radius),
            "ranking": [
                "total_pnl_desc",
                "itm_asc",
                "drawdown_desc",
                "trades_desc",
            ],
            "rows": out_rows,
        }

        out_path = Path(self.args.final_results_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"[final] wrote json path={out_path} rows={len(out_rows)}", flush=True)

    def run(self) -> None:
        print(
            f"[run] symbol={self.context.symbol} side={self.context.side} sobol_samples={self.args.sobol_samples} "
            f"workers={self.workers} parquet={self.parquet_path}",
            flush=True,
        )

        self._run_phase1()
        seed_rows = self._run_phase2()
        self._run_gradient(seed_rows)

        self._maybe_checkpoint(force=True)
        self._log_progress(force=True, prefix="run")
        self._write_final_json(seed_rows)


def _symbol_side_scoped_path(base_path: str, *, symbol: str, side: str) -> str:
    """Return a per-symbol/per-side output path derived from a shared default path.

    Example:
        data/sobol_gradient_trials.parquet -> data/sobol_gradient_trials_put.GDX.parquet
    """
    path = Path(base_path)
    clean_symbol = str(symbol).upper().strip()
    clean_side = str(side).lower().strip()
    return str(path.with_name(f"{path.stem}_{clean_side}.{clean_symbol}{path.suffix}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-phase Sobol + cached gradient backtest orchestration.")

    parser.add_argument("--features-parquet", default="data/features/option_strategy_features.parquet")
    parser.add_argument("--window-config-yaml", default="data/option_feature_windows.yaml")
    parser.add_argument(
        "--trials-parquet",
        default=None,
        help=(
            "Trials parquet path. If omitted, a symbol/side-specific path is used, "
            "for example data/sobol_gradient_trials_put.GDX.parquet."
        ),
    )
    parser.add_argument(
        "--final-results-json",
        default=None,
        help=(
            "Final JSON path. If omitted, a symbol/side-specific path is used, "
            "for example data/sobol_gradient_top_runs_put.GDX.json."
        ),
    )

    parser.add_argument("--symbol", required=True)
    parser.add_argument("--side", required=True, choices=["put", "call"])
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)

    parser.add_argument("--sobol-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument("--min-seed-trades", type=float, default=3.0)
    parser.add_argument("--max-seed-itm", type=float, default=2.0)
    parser.add_argument("--seed-top-ratio", type=float, default=0.30)
    parser.add_argument(
        "--seed-phases",
        default="phase1",
        help=(
            "Comma-separated trial phases eligible to seed phase 2. Use 'all' to allow prior phase2/gradient "
            "rows to become new local-search seeds."
        ),
    )
    parser.add_argument("--local-probe-per-seed", type=int, default=100)
    parser.add_argument("--local-radius", type=float, default=0.08)

    parser.add_argument("--gradient-steps", type=int, default=6)
    parser.add_argument("--gradient-step-size", type=float, default=0.03)
    parser.add_argument("--gradient-learning-rate", type=float, default=0.45)

    parser.add_argument("--progress-seconds", type=float, default=5.0)
    parser.add_argument("--checkpoint-seconds", type=float, default=5.0)
    parser.add_argument("--final-top-n", type=int, default=100)

    parser.add_argument("--roc-window-default", type=int, default=2)
    parser.add_argument("--vol-window-default", type=int, default=2)
    parser.add_argument("--roc-comparator", choices=["above", "below"], default="below")
    parser.add_argument("--vol-comparator", choices=["above", "below"], default="above")
    parser.add_argument("--roc-threshold-default", type=float, default=0.0)
    parser.add_argument("--vol-threshold-default", type=float, default=0.0)
    parser.add_argument("--roc-range-enabled", type=int, default=0)
    parser.add_argument("--vol-range-enabled", type=int, default=0)
    parser.add_argument("--roc-range-low", type=float, default=0.0)
    parser.add_argument("--roc-range-high", type=float, default=0.0)
    parser.add_argument("--vol-range-low", type=float, default=0.0)
    parser.add_argument("--vol-range-high", type=float, default=0.0)

    parser.add_argument("--risk-free-rate", type=float, default=0.04)
    parser.add_argument("--min-pricing-vol", type=float, default=0.10)
    parser.add_argument("--contract-size", type=int, default=100)

    args = parser.parse_args()

    args.symbol = str(args.symbol).upper().strip()
    args.side = str(args.side).lower().strip()

    if args.trials_parquet is None:
        args.trials_parquet = _symbol_side_scoped_path(
            "data/sobol_gradient_trials.parquet",
            symbol=args.symbol,
            side=args.side,
        )
    if args.final_results_json is None:
        args.final_results_json = _symbol_side_scoped_path(
            "data/sobol_gradient_top_runs.json",
            symbol=args.symbol,
            side=args.side,
        )

    if args.sobol_samples <= 0:
        raise ValueError("--sobol-samples must be > 0")
    if not (0.0 < float(args.seed_top_ratio) <= 1.0):
        raise ValueError("--seed-top-ratio must be in (0, 1]")
    if args.local_probe_per_seed < 0:
        raise ValueError("--local-probe-per-seed must be >= 0")
    if args.max_seed_itm < 0:
        raise ValueError("--max-seed-itm must be >= 0")
    if not (0.0 <= float(args.local_radius) <= 1.0):
        raise ValueError("--local-radius must be in [0, 1]")
    if args.gradient_steps < 0:
        raise ValueError("--gradient-steps must be >= 0")
    if args.final_top_n < 0:
        raise ValueError("--final-top-n must be >= 0")

    return args


def main() -> None:
    args = parse_args()
    orchestrator = Orchestrator(args)
    orchestrator.run()


if __name__ == "__main__":
    main()
