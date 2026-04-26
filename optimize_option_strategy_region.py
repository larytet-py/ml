#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from backtest_weekly_option_reversal import run_backtest
from optimization.constrained_bo import (
    ConstrainedBayesianOptimizer,
    ParamSpec,
    extract_region_summary,
    write_region_summary,
)
from optimization.goal_registry import DEFAULT_GOALS, build_goals
from weekly_option_backtest_common import summarize_trades


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
        "accel_roc_window": ParamSpec(low=args.accel_roc_window_min, high=args.accel_roc_window_max, is_int=True),
        "accel_shift_window": ParamSpec(low=args.accel_shift_window_min, high=args.accel_shift_window_max, is_int=True),
        "accel_ema_window": ParamSpec(low=args.accel_ema_window_min, high=args.accel_ema_window_max, is_int=True),
        "vol_window": ParamSpec(low=args.vol_window_min, high=args.vol_window_max, is_int=True),
        "vol_ema_window": ParamSpec(low=args.vol_ema_window_min, high=args.vol_ema_window_max, is_int=True),
        "corr_window": ParamSpec(low=args.corr_window_min, high=args.corr_window_max, is_int=True),
    }

    if side == "put":
        specs["roc_threshold"] = ParamSpec(low=args.put_roc_threshold_min, high=args.put_roc_threshold_max)
        specs["vol_threshold"] = ParamSpec(low=args.downside_vol_threshold_min, high=args.downside_vol_threshold_max)
        specs["accel_threshold"] = ParamSpec(low=args.put_accel_threshold_min, high=args.put_accel_threshold_max)
    else:
        specs["roc_threshold"] = ParamSpec(low=args.call_roc_threshold_min, high=args.call_roc_threshold_max)
        specs["vol_threshold"] = ParamSpec(low=args.upside_vol_threshold_min, high=args.upside_vol_threshold_max)
        specs["accel_threshold"] = ParamSpec(low=args.call_accel_threshold_min, high=args.call_accel_threshold_max)

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
        return metrics

    return evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constrained Bayesian optimization for weekly option strategy with region-flatness diagnostics.")
    parser.add_argument("--features-parquet", default="data/bayesian/option_strategy_features.parquet")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--side", choices=["put", "call"], required=True)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)

    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--random-init", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-pool-size", type=int, default=512)
    parser.add_argument("--constraint-penalty", type=float, default=1000.0)

    parser.add_argument("--trials-parquet", default="data/bo_trials.parquet")
    parser.add_argument("--eval-cache-parquet", default="data/bo_eval_cache.parquet")
    parser.add_argument("--region-summary-json", default="data/bo_region_summary.json")
    parser.add_argument("--goals-json", default=None, help="JSON list for up to 3 goals. If omitted, defaults are used.")

    parser.add_argument("--performance-tolerance-pct", type=float, default=2.0)
    parser.add_argument("--min-component-fraction", type=float, default=0.15)

    # Window bounds
    parser.add_argument("--roc-window-min", type=int, default=1)
    parser.add_argument("--roc-window-max", type=int, default=30)
    parser.add_argument("--accel-roc-window-min", type=int, default=1)
    parser.add_argument("--accel-roc-window-max", type=int, default=20)
    parser.add_argument("--accel-shift-window-min", type=int, default=1)
    parser.add_argument("--accel-shift-window-max", type=int, default=10)
    parser.add_argument("--accel-ema-window-min", type=int, default=2)
    parser.add_argument("--accel-ema-window-max", type=int, default=20)
    parser.add_argument("--vol-window-min", type=int, default=5)
    parser.add_argument("--vol-window-max", type=int, default=60)
    parser.add_argument("--vol-ema-window-min", type=int, default=2)
    parser.add_argument("--vol-ema-window-max", type=int, default=20)
    parser.add_argument("--corr-window-min", type=int, default=5)
    parser.add_argument("--corr-window-max", type=int, default=90)

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

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = _load_symbol_features(args.features_parquet, args.symbol, args.start_date, args.end_date)

    goals_cfg: List[Dict[str, object]]
    if args.goals_json:
        goals_cfg = json.loads(args.goals_json)
    else:
        goals_cfg = DEFAULT_GOALS
    goals = build_goals(goals_cfg)

    search_space = _build_search_space(args, side=args.side)
    evaluator = _evaluate_candidate_factory(df=df, args=args, side=args.side)

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
