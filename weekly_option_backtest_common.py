#!/usr/bin/env python3
import os
import random
import shlex
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


SCORE_TOLERANCE = 1e-12
DRAWDOWN_ZERO_TOLERANCE = 1e-12
INFEASIBLE_SCORE = -1_000_000_000.0


@dataclass
class OptimizationResult:
    params: Dict[str, float]
    score: float
    trades_df: pd.DataFrame
    metrics: Optional[Dict[str, float]]


def default_worker_count() -> int:
    return os.cpu_count() or 1


def shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def clip_value(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_worker_count(workers: int) -> int:
    if workers == 0:
        workers = os.cpu_count() or 1
    if workers < 0:
        workers = max(1, (os.cpu_count() or 1) + workers + 1)
    if workers < 1:
        raise ValueError("--workers must be >= 1, or 0 to use all CPUs, or negative to reserve cores.")
    return workers


def build_start_vectors(
    initial_vector: List[float],
    bounds: List[Tuple[float, float]],
    restarts: int,
    seed: int,
) -> List[List[float]]:
    rng = random.Random(seed)
    starts: List[List[float]] = [[clip_value(v, lo, hi) for v, (lo, hi) in zip(initial_vector, bounds)]]
    for _ in range(max(0, restarts - 1)):
        starts.append([rng.uniform(lo, hi) for lo, hi in bounds])
    return starts


def load_symbol_data(csv_path: str, symbol: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["symbol"] == symbol].copy()
    if df.empty:
        raise ValueError(f"No rows found for symbol '{symbol}' in {csv_path}")

    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "close", "high", "low", "volume", "dividend_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows left after date filters.")
    return df


def summarize_trades(trades_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    if trades_df.empty:
        return None

    total = len(trades_df)
    wins = int((trades_df["pnl_per_share"] > 0).sum())
    itm_expiries = int(trades_df["expired_itm"].sum())
    total_pnl = trades_df["pnl_per_contract"].sum()
    avg_pnl = trades_df["pnl_per_contract"].mean()
    median_pnl = trades_df["pnl_per_contract"].median()
    avg_return_on_spot = (trades_df["pnl_per_share"] / trades_df["entry_close"]).mean()

    equity = trades_df["pnl_per_contract"].cumsum()
    running_peak = equity.cummax()
    max_drawdown = (equity - running_peak).min()

    return {
        "total": float(total),
        "wins": float(wins),
        "win_rate": float(wins / total),
        "itm_expiries": float(itm_expiries),
        "itm_rate": float(itm_expiries / total),
        "total_pnl": float(total_pnl),
        "avg_pnl": float(avg_pnl),
        "median_pnl": float(median_pnl),
        "avg_return_on_spot": float(avg_return_on_spot),
        "max_drawdown": float(max_drawdown),
    }


def print_summary(trades_df: pd.DataFrame) -> None:
    metrics = summarize_trades(trades_df)
    if metrics is None:
        print("No trades generated with current settings.")
        return

    print(f"Trades: {int(metrics['total'])}")
    print(f"Win rate: {metrics['win_rate']:.2%}")
    print(f"Expired ITM: {int(metrics['itm_expiries'])} ({metrics['itm_rate']:.2%})")
    print(f"Total PnL (per 1 contract): {metrics['total_pnl']:.2f}")
    print(f"Average PnL/trade (per 1 contract): {metrics['avg_pnl']:.2f}")
    print(f"Median PnL/trade (per 1 contract): {metrics['median_pnl']:.2f}")
    print(f"Average return on spot notional: {metrics['avg_return_on_spot']:.4%}")
    print(f"Max drawdown (per 1 contract, cumulative): {metrics['max_drawdown']:.2f}")


def score_from_metrics(
    metrics: Optional[Dict[str, float]],
    min_trades: int,
    trade_penalty: float,
    goal_function: str,
) -> float:
    if metrics is None:
        return INFEASIBLE_SCORE

    total_trades = int(metrics["total"])
    shortfall = max(0, min_trades - total_trades)
    if shortfall > 0:
        return (
            -1_000_000.0
            - trade_penalty * float(shortfall)
            + 0.01 * float(total_trades)
            + 0.001 * float(metrics["avg_pnl"])
        )
    if goal_function == "min-itm-expiration":
        return -float(metrics["itm_expiries"])
    return float(metrics["avg_pnl"])


def is_better_score(
    candidate_score: float,
    candidate_metrics: Optional[Dict[str, float]],
    incumbent_score: float,
    incumbent_metrics: Optional[Dict[str, float]],
    goal_function: str,
    *,
    use_min_itm_tiebreak: bool = False,
    accept_equal_without_tiebreak: bool = False,
    score_tolerance: float = SCORE_TOLERANCE,
    drawdown_zero_tolerance: float = DRAWDOWN_ZERO_TOLERANCE,
) -> bool:
    if candidate_score > incumbent_score + score_tolerance:
        return True

    if abs(candidate_score - incumbent_score) > score_tolerance:
        return False

    if use_min_itm_tiebreak and goal_function == "min-itm-expiration":
        if candidate_metrics is None or incumbent_metrics is None:
            return False
        candidate_drawdown = abs(float(candidate_metrics["max_drawdown"]))
        incumbent_drawdown = abs(float(incumbent_metrics["max_drawdown"]))
        if candidate_drawdown < incumbent_drawdown - score_tolerance:
            return True
        if abs(candidate_drawdown - incumbent_drawdown) > score_tolerance:
            return False
        if candidate_drawdown > drawdown_zero_tolerance or incumbent_drawdown > drawdown_zero_tolerance:
            return False
        candidate_avg_pnl = float(candidate_metrics["avg_pnl"])
        incumbent_avg_pnl = float(incumbent_metrics["avg_pnl"])
        return candidate_avg_pnl > incumbent_avg_pnl + score_tolerance

    if accept_equal_without_tiebreak:
        return True
    return False
