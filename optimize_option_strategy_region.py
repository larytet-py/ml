#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning

from backtest_weekly_option_reversal import run_backtest
from optimization.constrained_bo import (
    ConstrainedBayesianOptimizer,
    ParamSpec,
    extract_region_summary,
    write_region_summary,
)
from optimization.goal_registry import DEFAULT_GOALS, build_goals
from weekly_option_backtest_common import summarize_trades


DEFAULT_WINDOW_RANGES = {
    "roc_window": {"min": 2, "max": 60},
    "accel_roc_window": {"min": 1, "max": 20},
    "accel_shift_window": {"min": 1, "max": 10},
    "vol_window": {"min": 2, "max": 40},
    "corr_window": {"min": 5, "max": 90},
}


def _apply_window_bounds_from_yaml(args: argparse.Namespace) -> None:
    loaded_ranges: Dict[str, Dict[str, int]] = {}
    if args.window_config_yaml:
        config_path = Path(args.window_config_yaml)
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text()) or {}
            window_ranges = loaded.get("window_ranges", {})
            if isinstance(window_ranges, dict):
                for key, value in window_ranges.items():
                    if isinstance(value, dict):
                        loaded_ranges[str(key)] = value

    arg_map = {
        "roc_window": ("roc_window_min", "roc_window_max"),
        "accel_roc_window": ("accel_roc_window_min", "accel_roc_window_max"),
        "accel_shift_window": ("accel_shift_window_min", "accel_shift_window_max"),
        "vol_window": ("vol_window_min", "vol_window_max"),
        "corr_window": ("corr_window_min", "corr_window_max"),
    }

    for key, (min_attr, max_attr) in arg_map.items():
        range_cfg = loaded_ranges.get(key, DEFAULT_WINDOW_RANGES[key])
        default_min = int(range_cfg.get("min", DEFAULT_WINDOW_RANGES[key]["min"]))
        default_max = int(range_cfg.get("max", DEFAULT_WINDOW_RANGES[key]["max"]))

        if getattr(args, min_attr) is None:
            setattr(args, min_attr, default_min)
        if getattr(args, max_attr) is None:
            setattr(args, max_attr, default_max)

        if int(getattr(args, min_attr)) > int(getattr(args, max_attr)):
            raise ValueError(
                f"Invalid range for {key}: min {getattr(args, min_attr)} is greater than max {getattr(args, max_attr)}"
            )


def _suppress_known_gp_warning_noise() -> None:
    warnings.filterwarnings(
        "ignore",
        message="The optimal value found for dimension 0 of parameter .* is close to the specified lower bound 1e-05.*",
        category=ConvergenceWarning,
    )


def _resolve_symbol_window_from_etfs(etfs_parquet: str, symbol: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    source_path = Path(etfs_parquet)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Unable to infer default dates: ETF source parquet not found at '{etfs_parquet}'. "
            "Pass --start-date/--end-date explicitly or provide --etfs-parquet."
        )

    df = pd.read_parquet(source_path)
    symbol_col = next((c for c in ("symbol", "ticker") if c in df.columns), None)
    if symbol_col is None:
        raise ValueError(
            f"Unable to infer default dates from {etfs_parquet}: expected one of columns ['symbol', 'ticker']."
        )
    date_col = next((c for c in ("date", "datetime", "timestamp") if c in df.columns), None)
    if date_col is None:
        raise ValueError(
            f"Unable to infer default dates from {etfs_parquet}: expected one of columns ['date', 'datetime', 'timestamp']."
        )

    sdf = df[df[symbol_col].astype(str).str.upper() == symbol.upper()].copy()
    if sdf.empty:
        raise ValueError(f"Unable to infer default dates: no rows for symbol {symbol} in {etfs_parquet}")
    sdf[date_col] = pd.to_datetime(sdf[date_col], errors="coerce")
    sdf = sdf.dropna(subset=[date_col])
    if sdf.empty:
        raise ValueError(f"Unable to infer default dates: symbol {symbol} has no valid dates in {etfs_parquet}")

    end_ts = pd.Timestamp(sdf[date_col].max()).normalize()
    start_ts = (end_ts - pd.DateOffset(months=12)).normalize()
    return start_ts, end_ts


def _resolve_symbol_window_from_features(features_parquet: str, symbol: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    df = pd.read_parquet(features_parquet)
    if "symbol" not in df.columns or "date" not in df.columns:
        raise ValueError(
            f"Unable to infer default dates from {features_parquet}: expected both 'symbol' and 'date' columns."
        )
    sdf = df[df["symbol"].astype(str).str.upper() == symbol.upper()].copy()
    if sdf.empty:
        raise ValueError(f"Unable to infer default dates: no rows for symbol {symbol} in {features_parquet}")
    sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce")
    sdf = sdf.dropna(subset=["date"])
    if sdf.empty:
        raise ValueError(f"Unable to infer default dates: symbol {symbol} has no valid dates in {features_parquet}")
    end_ts = pd.Timestamp(sdf["date"].max()).normalize()
    start_ts = (end_ts - pd.DateOffset(months=12)).normalize()
    return start_ts, end_ts


def _resolve_effective_dates(args: argparse.Namespace) -> tuple[Optional[str], Optional[str]]:
    start_date = args.start_date
    end_date = args.end_date
    try:
        inferred_start, inferred_end = _resolve_symbol_window_from_etfs(args.etfs_parquet, args.symbol)
    except (FileNotFoundError, ValueError):
        inferred_start, inferred_end = _resolve_symbol_window_from_features(args.features_parquet, args.symbol)
    if end_date:
        parsed_end = pd.to_datetime(end_date, errors="coerce")
        if pd.isna(parsed_end):
            raise ValueError(f"Invalid --end-date value: {end_date}")
        inferred_end = pd.Timestamp(parsed_end).normalize()
    if start_date:
        parsed_start = pd.to_datetime(start_date, errors="coerce")
        if pd.isna(parsed_start):
            raise ValueError(f"Invalid --start-date/--stare-date value: {start_date}")
        inferred_start = pd.Timestamp(parsed_start).normalize()
    elif end_date:
        inferred_start = (inferred_end - pd.DateOffset(months=12)).normalize()

    max_lookback_start = (inferred_end - pd.DateOffset(months=12)).normalize()
    if inferred_start < max_lookback_start:
        inferred_start = max_lookback_start
    if inferred_start > inferred_end:
        raise ValueError(
            f"Invalid effective date range: start {inferred_start.date().isoformat()} "
            f"is after end {inferred_end.date().isoformat()}."
        )

    return inferred_start.date().isoformat(), inferred_end.date().isoformat()


def _load_symbol_features(features_parquet: str, symbol: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    df = pd.read_parquet(features_parquet)
    if "symbol" not in df.columns:
        raise ValueError(f"Expected 'symbol' column in {features_parquet}")
    if "date" not in df.columns:
        raise ValueError(f"Expected 'date' column in {features_parquet}")

    df["symbol"] = df["symbol"].astype(str).str.upper()
    sdf = df[df["symbol"] == symbol.upper()].copy()
    if sdf.empty:
        raise ValueError(f"No rows for symbol {symbol} in {features_parquet}")

    sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in sdf.columns:
            sdf[col] = pd.to_numeric(sdf[col], errors="coerce")
    sdf = sdf.dropna(subset=["date", "open", "high", "low", "close"])
    if start_date:
        sdf = sdf[sdf["date"] >= pd.to_datetime(start_date)]
    if end_date:
        sdf = sdf[sdf["date"] <= pd.to_datetime(end_date)]
    sdf = sdf.sort_values("date").reset_index(drop=True)
    return sdf


def _build_search_space(args: argparse.Namespace, side: str) -> Dict[str, ParamSpec]:
    specs: Dict[str, ParamSpec] = {
        "roc_window": ParamSpec(low=args.roc_window_min, high=args.roc_window_max, is_int=True),
        "vol_window": ParamSpec(low=args.vol_window_min, high=args.vol_window_max, is_int=True),
    }

    if side == "put":
        specs["roc_threshold"] = ParamSpec(low=args.put_roc_threshold_min, high=args.put_roc_threshold_max)
        specs["vol_threshold"] = ParamSpec(low=args.downside_vol_threshold_min, high=args.downside_vol_threshold_max)
    else:
        specs["roc_threshold"] = ParamSpec(low=args.call_roc_threshold_min, high=args.call_roc_threshold_max)
        specs["vol_threshold"] = ParamSpec(low=args.upside_vol_threshold_min, high=args.upside_vol_threshold_max)

    return specs


def _evaluate_candidate_factory(df: pd.DataFrame, args: argparse.Namespace, side: str):
    def evaluate(params: Dict[str, float]) -> Dict[str, float]:
        roc_lookback = int(round(params["roc_window"]))
        vol_window = int(round(params["vol_window"]))

        if side == "put":
            put_roc_threshold = float(params["roc_threshold"])
            call_roc_threshold = float(args.call_roc_threshold_default)
            downside_vol_threshold = float(params["vol_threshold"])
            upside_vol_threshold = float(args.upside_vol_threshold_default)
        else:
            put_roc_threshold = float(args.put_roc_threshold_default)
            call_roc_threshold = float(params["roc_threshold"])
            downside_vol_threshold = float(args.downside_vol_threshold_default)
            upside_vol_threshold = float(params["vol_threshold"])

        trades_df = run_backtest(
            df=df,
            side=side,
            roc_lookback=roc_lookback,
            put_roc_threshold=put_roc_threshold,
            call_roc_threshold=call_roc_threshold,
            vol_window=vol_window,
            downside_vol_threshold_annualized=downside_vol_threshold,
            upside_vol_threshold_annualized=upside_vol_threshold,
            risk_free_rate=args.risk_free_rate,
            min_pricing_vol_annualized=args.min_pricing_vol,
            contract_size=args.contract_size,
            allow_overlap=args.allow_overlap,
            signal_df=None,
        )
        metrics = summarize_trades(trades_df)
        if metrics is None:
            return {
                "total": 0.0,
                "wins": 0.0,
                "win_rate": 0.0,
                "itm_expiries": float(args.no_trade_itm_penalty),
                "itm_rate": 1.0,
                "total_pnl": float(args.no_trade_pnl_penalty),
                "avg_pnl": float(args.no_trade_pnl_penalty),
                "median_pnl": float(args.no_trade_pnl_penalty),
                "avg_return_on_spot": 0.0,
                "max_drawdown": float(args.no_trade_drawdown_penalty),
            }
        if float(metrics.get("total", 0.0)) < float(args.min_trades):
            gated = dict(metrics)
            gated["itm_expiries"] = float(args.no_trade_itm_penalty)
            gated["itm_rate"] = 1.0
            gated["total_pnl"] = float(args.no_trade_pnl_penalty)
            gated["avg_pnl"] = float(args.no_trade_pnl_penalty)
            gated["median_pnl"] = float(args.no_trade_pnl_penalty)
            gated["max_drawdown"] = float(args.no_trade_drawdown_penalty)
            return gated
        return metrics

    return evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constrained Bayesian optimization for weekly option strategy with region-flatness diagnostics.")
    parser.add_argument("--features-parquet", default="data/bayesian/option_strategy_features.parquet")
    parser.add_argument("--window-config-yaml", default="data/bayesian/option_feature_windows.yaml")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--side", choices=["put", "call"], required=True)
    parser.add_argument("--start-date", "--stare-date", dest="start_date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--etfs-parquet", default="data/etfs.parquet")
    parser.add_argument("--min-trades", type=int, default=6)

    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--random-init", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-pool-size", type=int, default=512)
    parser.add_argument("--constraint-penalty", type=float, default=1000.0)
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore existing BO trials/eval cache files and rebuild results from scratch for this run.",
    )

    parser.add_argument("--trials-parquet", default="data/bo_trials.parquet")
    parser.add_argument("--eval-cache-parquet", default="data/bo_eval_cache.parquet")
    parser.add_argument("--region-summary-json", default="data/bo_region_summary.json")
    parser.add_argument("--goals-json", default=None, help="JSON list for up to 3 goals. If omitted, defaults are used.")

    parser.add_argument("--performance-tolerance-pct", type=float, default=2.0)
    parser.add_argument("--min-component-fraction", type=float, default=0.15)

    # Window bounds
    parser.add_argument("--roc-window-min", type=int, default=None)
    parser.add_argument("--roc-window-max", type=int, default=None)
    parser.add_argument("--accel-roc-window-min", type=int, default=None)
    parser.add_argument("--accel-roc-window-max", type=int, default=None)
    parser.add_argument("--accel-shift-window-min", type=int, default=None)
    parser.add_argument("--accel-shift-window-max", type=int, default=None)
    parser.add_argument("--accel-ema-window-min", type=int, default=2)
    parser.add_argument("--accel-ema-window-max", type=int, default=20)
    parser.add_argument("--vol-window-min", type=int, default=None)
    parser.add_argument("--vol-window-max", type=int, default=None)
    parser.add_argument("--vol-ema-window-min", type=int, default=2)
    parser.add_argument("--vol-ema-window-max", type=int, default=20)
    parser.add_argument("--corr-window-min", type=int, default=None)
    parser.add_argument("--corr-window-max", type=int, default=None)

    # Threshold bounds
    parser.add_argument("--put-roc-threshold-min", type=float, default=-0.40)
    parser.add_argument("--put-roc-threshold-max", type=float, default=0.40)
    parser.add_argument("--call-roc-threshold-min", type=float, default=-0.40)
    parser.add_argument("--call-roc-threshold-max", type=float, default=0.40)
    parser.add_argument("--downside-vol-threshold-min", type=float, default=0.00)
    parser.add_argument("--downside-vol-threshold-max", type=float, default=1.00)
    parser.add_argument("--upside-vol-threshold-min", type=float, default=0.00)
    parser.add_argument("--upside-vol-threshold-max", type=float, default=1.00)
    parser.add_argument("--put-accel-threshold-min", type=float, default=-0.20)
    parser.add_argument("--put-accel-threshold-max", type=float, default=0.20)
    parser.add_argument("--call-accel-threshold-min", type=float, default=-0.20)
    parser.add_argument("--call-accel-threshold-max", type=float, default=0.20)

    parser.add_argument("--put-roc-threshold-default", type=float, default=-0.03)
    parser.add_argument("--call-roc-threshold-default", type=float, default=0.03)
    parser.add_argument("--downside-vol-threshold-default", type=float, default=0.20)
    parser.add_argument("--upside-vol-threshold-default", type=float, default=0.20)

    parser.add_argument("--risk-free-rate", type=float, default=0.04)
    parser.add_argument("--min-pricing-vol", type=float, default=0.10)
    parser.add_argument("--contract-size", type=int, default=100)
    parser.add_argument("--allow-overlap", action="store_true")

    parser.add_argument("--no-trade-itm-penalty", type=float, default=1_000_000.0)
    parser.add_argument("--no-trade-pnl-penalty", type=float, default=-1_000_000.0)
    parser.add_argument("--no-trade-drawdown-penalty", type=float, default=1_000_000.0)
    parser.add_argument(
        "--bo-progress-seconds",
        type=float,
        default=5.0,
        help="How often to emit BO progress logs in seconds. Set <=0 to disable periodic logs.",
    )

    args = parser.parse_args()
    _apply_window_bounds_from_yaml(args)
    return args


def main() -> None:
    args = parse_args()
    _suppress_known_gp_warning_noise()
    start_date, end_date = _resolve_effective_dates(args)
    print(
        "[bo] run "
        f"symbol={args.symbol.upper()} "
        f"side={args.side} "
        f"date_range={start_date}..{end_date} "
        f"min_trades={int(args.min_trades)} "
        f"trials={int(args.trials)} "
        f"force_rebuild={bool(args.force_rebuild)}",
        flush=True,
    )
    df = _load_symbol_features(args.features_parquet, args.symbol, start_date, end_date)

    goals_cfg: List[Dict[str, object]]
    if args.goals_json:
        goals_cfg = json.loads(args.goals_json)
    else:
        goals_cfg = DEFAULT_GOALS
    goals = build_goals(goals_cfg)

    search_space = _build_search_space(args, side=args.side)
    evaluator = _evaluate_candidate_factory(df=df, args=args, side=args.side)
    param_name_aliases = {
        "roc_window": "roc-lookback",
        "vol_window": "vol-window",
        "roc_threshold": "put-roc-threshold" if args.side == "put" else "call-roc-threshold",
        "vol_threshold": "downside-vol-threshold" if args.side == "put" else "upside-vol-threshold",
    }

    optimizer = ConstrainedBayesianOptimizer(
        search_space=search_space,
        goals=goals,
        evaluator=evaluator,
        trials_parquet=args.trials_parquet,
        eval_cache_parquet=args.eval_cache_parquet,
        seed=args.seed,
        n_random_init=args.random_init,
        candidate_pool_size=args.candidate_pool_size,
        constraint_penalty=args.constraint_penalty,
        progress_interval_seconds=args.bo_progress_seconds,
        force_rebuild=args.force_rebuild,
        param_name_aliases=param_name_aliases,
    )
    trials = optimizer.run(n_trials=args.trials)

    summary = extract_region_summary(
        trials=trials,
        search_space=search_space,
        performance_tolerance_pct=args.performance_tolerance_pct,
        min_component_fraction=args.min_component_fraction,
    )
    write_region_summary(args.region_summary_json, summary)

    feasible = [t for t in trials if t.goals.feasible]
    if feasible:
        best = max(feasible, key=lambda t: t.goals.objective_score)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "trial_count": len(trials),
                    "feasible_count": len(feasible),
                    "best_trial_id": best.trial_id,
                    "best_params": best.params,
                    "best_metrics": best.metrics,
                    "effective_start_date": start_date,
                    "effective_end_date": end_date,
                    "min_trades": int(args.min_trades),
                    "force_rebuild": bool(args.force_rebuild),
                    "region_summary_json": str(Path(args.region_summary_json)),
                    "trials_parquet": str(Path(args.trials_parquet)),
                    "eval_cache_parquet": str(Path(args.eval_cache_parquet)),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "status": "no_feasible_trials",
                    "trial_count": len(trials),
                    "effective_start_date": start_date,
                    "effective_end_date": end_date,
                    "min_trades": int(args.min_trades),
                    "force_rebuild": bool(args.force_rebuild),
                    "region_summary_json": str(Path(args.region_summary_json)),
                    "trials_parquet": str(Path(args.trials_parquet)),
                    "eval_cache_parquet": str(Path(args.eval_cache_parquet)),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
