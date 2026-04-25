#!/usr/bin/env python3
"""
Usage:

SELECT
    api_security.symbol,
    api_securitydailydata.date,
    api_securitydailydata.open,
    api_securitydailydata.close,
    api_securitydailydata.high,
    api_securitydailydata.low,
    api_securitydailydata.volume,
    api_securitydailydata.dividend_rate
FROM api_securitydailydata
INNER JOIN api_security
    ON api_securitydailydata.security_id = api_security.id
WHERE api_security.active = TRUE
  AND (
        api_security.etf = TRUE
        OR api_security.symbol = 'VXX'
      )
  AND api_securitydailydata.date >= CURRENT_DATE - INTERVAL '12 months'
ORDER BY api_security.symbol, api_securitydailydata.date;

python3 backtest_weekly_option_reversal.py \
  --symbol SPY \
  --side put \
  --optimize
"""
import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from option_pricing import BlackScholesPricer
from weekly_option_backtest_common import (
    OptimizationResult,
    build_start_vectors,
    default_worker_count,
    is_better_score,
    load_symbol_data,
    normalize_worker_count,
    score_from_metrics,
    shell_join,
    summarize_trades,
)
from weekly_option_output import (
    build_output_line,
    print_optimization_result,
    print_recent_trades,
    print_summary_with_output_line,
)
from weekly_option_reversal_core import build_signal_frame, compute_weekly_entry_candidates


def _format_best_backtest_command(
    script_path: str,
    symbol: str,
    side: str,
    params: Dict[str, float],
    csv_path: str,
    start_date: Optional[str],
    end_date: Optional[str],
    allow_overlap: bool,
) -> str:
    cmd = [
        script_path,
        "--symbol",
        symbol,
        "--side",
        side,
        "--roc-lookback",
        str(int(params["roc_lookback"])),
        "--vol-window",
        str(int(params["vol_window"])),
    ]
    if side == "put":
        cmd.extend(
            [
                "--put-roc-threshold",
                f"{params['put_roc_threshold']:.6f}",
                "--downside-vol-threshold",
                f"{params['downside_vol_threshold_annualized']:.6f}",
            ]
        )
    else:
        cmd.extend(
            [
                "--call-roc-threshold",
                f"{params['call_roc_threshold']:.6f}",
                "--upside-vol-threshold",
                f"{params['upside_vol_threshold_annualized']:.6f}",
            ]
        )
    if csv_path != "data/etfs-all.csv":
        cmd.extend(["--csv", csv_path])
    if start_date:
        cmd.extend(["--start-date", start_date])
    if end_date:
        cmd.extend(["--end-date", end_date])
    if allow_overlap:
        cmd.append("--allow-overlap")
    return shell_join(cmd)


def _format_best_backtest_flags(
    symbol: str,
    side: str,
    params: Dict[str, float],
    csv_path: str,
    start_date: Optional[str],
    end_date: Optional[str],
    allow_overlap: bool,
) -> str:
    flags = [
        "--symbol",
        symbol,
        "--side",
        side,
        "--roc-lookback",
        str(int(params["roc_lookback"])),
        "--vol-window",
        str(int(params["vol_window"])),
    ]
    if side == "put":
        flags.extend(
            [
                "--put-roc-threshold",
                f"{params['put_roc_threshold']:.6f}",
                "--downside-vol-threshold",
                f"{params['downside_vol_threshold_annualized']:.6f}",
            ]
        )
    else:
        flags.extend(
            [
                "--call-roc-threshold",
                f"{params['call_roc_threshold']:.6f}",
                "--upside-vol-threshold",
                f"{params['upside_vol_threshold_annualized']:.6f}",
            ]
        )
    if csv_path != "data/etfs-all.csv":
        flags.extend(["--csv", csv_path])
    if start_date:
        flags.extend(["--start-date", start_date])
    if end_date:
        flags.extend(["--end-date", end_date])
    if allow_overlap:
        flags.append("--allow-overlap")
    return shell_join(flags)


@dataclass
class Trade:
    side: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_close: float
    exit_close: float
    strike: float
    premium: float
    intrinsic_at_expiry: float
    expired_itm: bool
    pnl_per_share: float
    pnl_per_contract: float
    roc_signal: float
    trend_vol_signal: float
    pricing_vol: float
    scheduled_expiry_date: pd.Timestamp
    time_to_expiry_days: int


def run_backtest(
    df: pd.DataFrame,
    side: str,
    roc_lookback: int,
    put_roc_threshold: float,
    call_roc_threshold: float,
    vol_window: int,
    downside_vol_threshold_annualized: float,
    upside_vol_threshold_annualized: float,
    risk_free_rate: float,
    min_pricing_vol_annualized: float,
    contract_size: int,
    allow_overlap: bool,
    signal_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    pricer = BlackScholesPricer(
        risk_free_rate=risk_free_rate,
        min_sigma=min_pricing_vol_annualized,
    )

    if signal_df is None:
        signal_df = build_signal_frame(df, roc_lookback=roc_lookback, vol_window=vol_window)
    entries = compute_weekly_entry_candidates(
        signal_df=signal_df,
        side=side,
        put_roc_threshold=put_roc_threshold,
        call_roc_threshold=call_roc_threshold,
        downside_vol_threshold_annualized=downside_vol_threshold_annualized,
        upside_vol_threshold_annualized=upside_vol_threshold_annualized,
        allow_overlap=allow_overlap,
    )
    dates = signal_df["date"].to_numpy()
    close = signal_df["close"].to_numpy(dtype=float, copy=False)
    pricing_vol_arr = signal_df["pricing_vol_annualized"].to_numpy(dtype=float, copy=False)
    roc = signal_df["roc"].to_numpy(dtype=float, copy=False)

    trades = []
    for i, entry in entries.items():
        entry_close = float(close[i])
        entry_date = pd.Timestamp(dates[i])
        strike = entry_close
        exit_idx = int(entry["exit_idx"])
        days_to_friday = int(entry["days_to_friday"])
        trend_vol_signal = float(entry["trend_vol_signal"])
        scheduled_expiry_date = (entry_date + pd.Timedelta(days=days_to_friday)).normalize()
        time_to_expiry_days = days_to_friday
        time_to_expiry_years = time_to_expiry_days / 365.25

        if exit_idx >= len(signal_df):
            continue

        raw_pricing_vol = float(pricing_vol_arr[i])
        used_pricing_vol = pricer.effective_sigma(raw_pricing_vol)
        premium = pricer.price(
            side=side,
            spot=entry_close,
            strike=strike,
            time_to_expiry_years=time_to_expiry_years,
            sigma=raw_pricing_vol,
        )

        exit_close = float(close[exit_idx])
        intrinsic = pricer.intrinsic_value(side=side, strike=strike, spot=exit_close)
        expired_itm = intrinsic > 0.0
        pnl_per_share = premium - intrinsic

        trades.append(
            Trade(
                side=side,
                entry_date=pd.Timestamp(dates[i]),
                exit_date=pd.Timestamp(dates[exit_idx]),
                entry_close=entry_close,
                exit_close=exit_close,
                strike=strike,
                premium=premium,
                intrinsic_at_expiry=intrinsic,
                expired_itm=expired_itm,
                pnl_per_share=pnl_per_share,
                pnl_per_contract=pnl_per_share * contract_size,
                roc_signal=float(roc[i]),
                trend_vol_signal=trend_vol_signal,
                pricing_vol=used_pricing_vol,
                scheduled_expiry_date=scheduled_expiry_date,
                time_to_expiry_days=time_to_expiry_days,
            )
        )

    return pd.DataFrame([t.__dict__ for t in trades])


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _build_params_from_vector(
    x: List[float],
    side: str,
    fixed_put_roc_threshold: float,
    fixed_call_roc_threshold: float,
    fixed_downside_vol_threshold: float,
    fixed_upside_vol_threshold: float,
) -> Dict[str, float]:
    roc_lookback = int(round(x[0]))
    roc_threshold = float(x[1])
    vol_window = int(round(x[2]))
    vol_threshold = float(x[3])

    params = {
        "roc_lookback": roc_lookback,
        "put_roc_threshold": fixed_put_roc_threshold,
        "call_roc_threshold": fixed_call_roc_threshold,
        "vol_window": vol_window,
        "downside_vol_threshold_annualized": fixed_downside_vol_threshold,
        "upside_vol_threshold_annualized": fixed_upside_vol_threshold,
    }
    if side == "put":
        params["put_roc_threshold"] = roc_threshold
        params["downside_vol_threshold_annualized"] = vol_threshold
    else:
        params["call_roc_threshold"] = roc_threshold
        params["upside_vol_threshold_annualized"] = vol_threshold
    return params


def _optimize_single_start(
    df: pd.DataFrame,
    side: str,
    start: List[float],
    bounds: List[Tuple[float, float]],
    iterations: int,
    learning_rate: float,
    finite_diff_eps: float,
    min_trades: int,
    trade_penalty: float,
    risk_free_rate: float,
    min_pricing_vol_annualized: float,
    contract_size: int,
    allow_overlap: bool,
    fixed_put_roc_threshold: float,
    fixed_call_roc_threshold: float,
    fixed_downside_vol_threshold: float,
    fixed_upside_vol_threshold: float,
    goal_function: str,
) -> Tuple[OptimizationResult, int]:
    vector_len = len(start)
    int_dims = {0, 2}
    signal_cache: Dict[Tuple[int, int], pd.DataFrame] = {}

    def project(x: List[float]) -> List[float]:
        return [_clip(v, lo, hi) for v, (lo, hi) in zip(x, bounds)]

    def evaluate(x: List[float]) -> Tuple[float, pd.DataFrame, Optional[Dict[str, float]], Dict[str, float]]:
        params = _build_params_from_vector(
            x=project(x),
            side=side,
            fixed_put_roc_threshold=fixed_put_roc_threshold,
            fixed_call_roc_threshold=fixed_call_roc_threshold,
            fixed_downside_vol_threshold=fixed_downside_vol_threshold,
            fixed_upside_vol_threshold=fixed_upside_vol_threshold,
        )
        roc_lookback = int(params["roc_lookback"])
        vol_window = int(params["vol_window"])
        signal_key = (roc_lookback, vol_window)
        signal_df = signal_cache.get(signal_key)
        if signal_df is None:
            signal_df = build_signal_frame(df, roc_lookback=roc_lookback, vol_window=vol_window)
            signal_cache[signal_key] = signal_df

        trades_df = run_backtest(
            df=df,
            side=side,
            roc_lookback=roc_lookback,
            put_roc_threshold=float(params["put_roc_threshold"]),
            call_roc_threshold=float(params["call_roc_threshold"]),
            vol_window=vol_window,
            downside_vol_threshold_annualized=float(params["downside_vol_threshold_annualized"]),
            upside_vol_threshold_annualized=float(params["upside_vol_threshold_annualized"]),
            risk_free_rate=risk_free_rate,
            min_pricing_vol_annualized=min_pricing_vol_annualized,
            contract_size=contract_size,
            allow_overlap=allow_overlap,
            signal_df=signal_df,
        )
        metrics = summarize_trades(trades_df)
        score = score_from_metrics(
            metrics=metrics,
            min_trades=min_trades,
            trade_penalty=trade_penalty,
            goal_function=goal_function,
        )
        return score, trades_df, metrics, params

    def finite_difference_gradient(x: List[float]) -> List[float]:
        grad = []
        for j in range(vector_len):
            lo, hi = bounds[j]
            width = hi - lo
            if j in int_dims:
                step = 1.0
            else:
                step = max(width * finite_diff_eps, 1e-6)
            x_hi = x.copy()
            x_lo = x.copy()
            x_hi[j] = _clip(x_hi[j] + step, lo, hi)
            x_lo[j] = _clip(x_lo[j] - step, lo, hi)

            if abs(x_hi[j] - x_lo[j]) < 1e-12:
                grad.append(0.0)
                continue

            s_hi, _, _, _ = evaluate(x_hi)
            s_lo, _, _, _ = evaluate(x_lo)
            grad.append((s_hi - s_lo) / (x_hi[j] - x_lo[j]))
        return grad

    x = project(start.copy())
    score, trades_df, metrics, params = evaluate(x)
    eval_count = 1

    for _ in range(iterations):
        grad = finite_difference_gradient(x)
        norm = math.sqrt(sum(g * g for g in grad))
        if norm <= 1e-12:
            break

        alpha = learning_rate
        accepted = False
        while alpha >= 1e-3:
            candidate = []
            for j in range(vector_len):
                lo, hi = bounds[j]
                span = hi - lo
                delta = alpha * (grad[j] / norm) * span
                candidate.append(_clip(x[j] + delta, lo, hi))

            candidate_score, candidate_df, candidate_metrics, candidate_params = evaluate(candidate)
            eval_count += 1
            if is_better_score(
                candidate_score=candidate_score,
                candidate_metrics=candidate_metrics,
                incumbent_score=score,
                incumbent_metrics=metrics,
                goal_function=goal_function,
                use_min_itm_tiebreak=True,
            ):
                x = candidate
                score = candidate_score
                trades_df = candidate_df
                metrics = candidate_metrics
                params = candidate_params
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            break

    return (
        OptimizationResult(params=params, score=score, trades_df=trades_df, metrics=metrics),
        eval_count,
    )


def optimize_parameters(
    df: pd.DataFrame,
    side: str,
    symbol: str,
    script_path: str,
    csv_path: str,
    start_date: Optional[str],
    end_date: Optional[str],
    initial_vector: List[float],
    bounds: List[Tuple[float, float]],
    iterations: int,
    restarts: int,
    learning_rate: float,
    finite_diff_eps: float,
    min_trades: int,
    trade_penalty: float,
    seed: int,
    risk_free_rate: float,
    min_pricing_vol_annualized: float,
    contract_size: int,
    allow_overlap: bool,
    fixed_put_roc_threshold: float,
    fixed_call_roc_threshold: float,
    fixed_downside_vol_threshold: float,
    fixed_upside_vol_threshold: float,
    goal_function: str,
    progress_interval_seconds: float,
    workers: int,
) -> OptimizationResult:
    if any(lo >= hi for lo, hi in bounds):
        raise ValueError("Invalid optimization bounds: each min value must be less than max.")
    workers = normalize_worker_count(workers)

    best_score = -1_000_000_000.0
    best_df = pd.DataFrame()
    best_metrics: Optional[Dict[str, float]] = None
    best_params: Dict[str, float] = {}
    eval_count = 0
    start_time = time.monotonic()
    last_report_time = start_time
    last_restart_report_time = start_time

    starts = build_start_vectors(
        initial_vector=initial_vector,
        bounds=bounds,
        restarts=restarts,
        seed=seed,
    )

    def report_best_progress() -> None:
        elapsed = time.monotonic() - start_time
        best_cmd = _format_best_backtest_command(
            script_path=script_path,
            symbol=symbol,
            side=side,
            params=best_params,
            csv_path=csv_path,
            start_date=start_date,
            end_date=end_date,
            allow_overlap=allow_overlap,
        )
        print(
            "[opt] "
            f"t+{elapsed:.1f}s "
            f"evals={eval_count} "
            f"goal={goal_function} "
            f"score={best_score:.6f} "
            f"avg_pnl={best_metrics['avg_pnl']:.2f} "
            f"itm_count={int(best_metrics['itm_expiries'])} "
            f"trades={int(best_metrics['total'])} "
            f"cmd={best_cmd}",
            flush=True,
        )

    def maybe_report_progress() -> None:
        nonlocal last_report_time
        if progress_interval_seconds <= 0:
            return
        now = time.monotonic()
        if now - last_report_time < progress_interval_seconds:
            return
        elapsed = now - start_time
        if best_metrics is None:
            print(f"[opt] t+{elapsed:.1f}s evals={eval_count} best_avg_pnl=n/a trades=0")
        else:
            report_best_progress()
        last_report_time = now

    def maybe_report_restart_progress(completed: int, total: int, max_workers: int, force: bool = False) -> None:
        nonlocal last_restart_report_time
        if progress_interval_seconds <= 0:
            return
        now = time.monotonic()
        if not force and now - last_restart_report_time < progress_interval_seconds:
            return
        elapsed = now - start_time
        print(
            "[opt] "
            f"t+{elapsed:.1f}s "
            f"restarts={completed}/{total} "
            f"evals={eval_count} "
            f"workers={max_workers}",
            flush=True,
        )
        last_restart_report_time = now

    if workers == 1 or len(starts) == 1:
        for start in starts:
            restart_result, restart_evals = _optimize_single_start(
                df=df,
                side=side,
                start=start,
                bounds=bounds,
                iterations=iterations,
                learning_rate=learning_rate,
                finite_diff_eps=finite_diff_eps,
                min_trades=min_trades,
                trade_penalty=trade_penalty,
                risk_free_rate=risk_free_rate,
                min_pricing_vol_annualized=min_pricing_vol_annualized,
                contract_size=contract_size,
                allow_overlap=allow_overlap,
                fixed_put_roc_threshold=fixed_put_roc_threshold,
                fixed_call_roc_threshold=fixed_call_roc_threshold,
                fixed_downside_vol_threshold=fixed_downside_vol_threshold,
                fixed_upside_vol_threshold=fixed_upside_vol_threshold,
                goal_function=goal_function,
            )
            eval_count += restart_evals
            if is_better_score(
                candidate_score=restart_result.score,
                candidate_metrics=restart_result.metrics,
                incumbent_score=best_score,
                incumbent_metrics=best_metrics,
                goal_function=goal_function,
                use_min_itm_tiebreak=True,
            ):
                best_score = restart_result.score
                best_df = restart_result.trades_df
                best_metrics = restart_result.metrics
                best_params = restart_result.params
                report_best_progress()
            maybe_report_progress()
    else:
        max_workers = min(workers, len(starts))
        completed = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _optimize_single_start,
                    df,
                    side,
                    start,
                    bounds,
                    iterations,
                    learning_rate,
                    finite_diff_eps,
                    min_trades,
                    trade_penalty,
                    risk_free_rate,
                    min_pricing_vol_annualized,
                    contract_size,
                    allow_overlap,
                    fixed_put_roc_threshold,
                    fixed_call_roc_threshold,
                    fixed_downside_vol_threshold,
                    fixed_upside_vol_threshold,
                    goal_function,
                )
                for start in starts
            ]
            for future in as_completed(futures):
                restart_result, restart_evals = future.result()
                completed += 1
                eval_count += restart_evals
                if is_better_score(
                    candidate_score=restart_result.score,
                    candidate_metrics=restart_result.metrics,
                    incumbent_score=best_score,
                    incumbent_metrics=best_metrics,
                    goal_function=goal_function,
                    use_min_itm_tiebreak=True,
                ):
                    best_score = restart_result.score
                    best_df = restart_result.trades_df
                    best_metrics = restart_result.metrics
                    best_params = restart_result.params
                    report_best_progress()
                maybe_report_restart_progress(completed=completed, total=len(starts), max_workers=max_workers)
            maybe_report_restart_progress(
                completed=completed,
                total=len(starts),
                max_workers=max_workers,
                force=True,
            )

    return OptimizationResult(params=best_params, score=best_score, trades_df=best_df, metrics=best_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest: sell ATM weekly puts/calls after trend shock (ROC + directional stddev trigger)."
    )
    parser.add_argument("--csv", default="data/etfs-all.csv", help="Path to input CSV.")
    parser.add_argument("--symbol", required=True, help="Ticker to backtest, e.g. SPY.")
    parser.add_argument(
        "--side",
        choices=["put", "call"],
        default="put",
        help="`put`: sell after downtrend. `call`: sell after uptrend.",
    )
    parser.add_argument("--start-date", default=None, help="Optional filter, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Optional filter, YYYY-MM-DD.")
    parser.add_argument("--roc-lookback", type=int, default=5, help="Days for ROC.")
    parser.add_argument("--put-roc-threshold", type=float, default=-0.25, help="Put trigger: ROC <= threshold.")
    parser.add_argument("--call-roc-threshold", type=float, default=0.25, help="Call trigger: ROC >= threshold.")
    parser.add_argument(
        "--vol-window",
        type=int,
        default=10,
        help="Rolling window for trigger volatility (downside/upside), not Black-Scholes pricing vol.",
    )
    parser.add_argument(
        "--downside-vol-threshold",
        type=float,
        default=0.05,
        help="Annualized downside vol threshold for put signal.",
    )
    parser.add_argument(
        "--upside-vol-threshold",
        type=float,
        default=0.05,
        help="Annualized upside vol threshold for call signal.",
    )
    parser.add_argument("--risk-free-rate", type=float, default=0.04, help="Risk-free rate for Black-Scholes.")
    parser.add_argument(
        "--min-pricing-vol",
        type=float,
        default=0.10,
        help="Vol floor (annualized) used when pricing the put premium.",
    )
    parser.add_argument("--contract-size", type=int, default=100, help="Shares per option contract.")
    parser.add_argument("--allow-overlap", action="store_true", help="Allow overlapping weekly positions.")
    parser.add_argument("--trades-out", default=None, help="Optional CSV path for trade log.")
    parser.add_argument(
        "--print-trades",
        type=int,
        default=10,
        help="How many recent trades to print. Use -1 to print all trades.",
    )
    parser.add_argument("--optimize", action="store_true", help="Run gradient-based optimization for strategy parameters.")
    parser.add_argument("--opt-iters", type=int, default=80, help="Iterations per optimization restart.")
    parser.add_argument("--opt-restarts", type=int, default=150, help="Number of optimization restarts.")
    parser.add_argument("--opt-learning-rate", type=float, default=0.25, help="Projected gradient ascent step size.")
    parser.add_argument("--opt-fd-eps", type=float, default=0.03, help="Finite-difference relative step for continuous params.")
    parser.add_argument(
        "--opt-min-trades",
        type=int,
        default=6,
        help="Minimum trades target used in objective penalty.",
    )
    parser.add_argument(
        "--opt-trade-penalty",
        type=float,
        default=75.0,
        help="Penalty points per missing trade below --opt-min-trades.",
    )
    parser.add_argument(
        "--opt-goal-function",
        choices=["avg-pnl", "min-itm-expiration"],
        default="avg-pnl",
        help="Optimization objective: maximize avg-pnl (default) or minimize ITM expirations.",
    )
    parser.add_argument("--opt-seed", type=int, default=7, help="Random seed for optimizer restarts.")
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Optimization worker processes. Default: all CPU cores. 1 disables multiprocessing, 0 uses all cores, negative reserves cores.",
    )
    parser.add_argument(
        "--opt-progress-seconds",
        type=float,
        default=1.0,
        help="How often to print optimization progress. Set <=0 to disable.",
    )
    parser.add_argument("--roc-lookback-min", type=int, default=2, help="Lower bound for roc-lookback in optimization.")
    parser.add_argument("--roc-lookback-max", type=int, default=60, help="Upper bound for roc-lookback in optimization.")
    parser.add_argument("--vol-window-min", type=int, default=2, help="Lower bound for vol-window in optimization.")
    parser.add_argument("--vol-window-max", type=int, default=40, help="Upper bound for vol-window in optimization.")
    parser.add_argument(
        "--roc-threshold-min",
        type=float,
        default=None,
        help="Lower bound for side-specific ROC threshold in optimization.",
    )
    parser.add_argument(
        "--roc-threshold-max",
        type=float,
        default=None,
        help="Upper bound for side-specific ROC threshold in optimization.",
    )
    parser.add_argument(
        "--vol-threshold-min",
        type=float,
        default=0.000,
        help="Lower bound for side-specific vol threshold in optimization.",
    )
    parser.add_argument(
        "--vol-threshold-max",
        type=float,
        default=1.00,
        help="Upper bound for side-specific vol threshold in optimization.",
    )
    args = parser.parse_args()
    script_name = Path(__file__).name

    df = load_symbol_data(args.csv, args.symbol, args.start_date, args.end_date)
    replay_flags: Optional[str] = None
    commented_output = bool(args.optimize)
    output_line = build_output_line(commented_output)

    if args.optimize:
        if args.side == "call":
            roc_threshold_min = args.roc_threshold_min if args.roc_threshold_min is not None else 0.00
            roc_threshold_max = args.roc_threshold_max if args.roc_threshold_max is not None else 0.20
            initial_roc_threshold = args.call_roc_threshold
            initial_vol_threshold = args.upside_vol_threshold
        else:
            roc_threshold_min = args.roc_threshold_min if args.roc_threshold_min is not None else -0.30
            roc_threshold_max = args.roc_threshold_max if args.roc_threshold_max is not None else 0.00
            initial_roc_threshold = args.put_roc_threshold
            initial_vol_threshold = args.downside_vol_threshold

        initial_vector = [
            float(args.roc_lookback),
            float(initial_roc_threshold),
            float(args.vol_window),
            float(initial_vol_threshold),
        ]
        bounds = [
            (float(args.roc_lookback_min), float(args.roc_lookback_max)),
            (float(roc_threshold_min), float(roc_threshold_max)),
            (float(args.vol_window_min), float(args.vol_window_max)),
            (float(args.vol_threshold_min), float(args.vol_threshold_max)),
        ]

        result = optimize_parameters(
            df=df,
            side=args.side,
            symbol=args.symbol,
            script_path=script_name,
            csv_path=args.csv,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_vector=initial_vector,
            bounds=bounds,
            iterations=args.opt_iters,
            restarts=args.opt_restarts,
            learning_rate=args.opt_learning_rate,
            finite_diff_eps=args.opt_fd_eps,
            min_trades=args.opt_min_trades,
            trade_penalty=args.opt_trade_penalty,
            seed=args.opt_seed,
            risk_free_rate=args.risk_free_rate,
            min_pricing_vol_annualized=args.min_pricing_vol,
            contract_size=args.contract_size,
            allow_overlap=args.allow_overlap,
            fixed_put_roc_threshold=args.put_roc_threshold,
            fixed_call_roc_threshold=args.call_roc_threshold,
            fixed_downside_vol_threshold=args.downside_vol_threshold,
            fixed_upside_vol_threshold=args.upside_vol_threshold,
            goal_function=args.opt_goal_function,
            progress_interval_seconds=args.opt_progress_seconds,
            workers=args.workers,
        )

        trades_df = result.trades_df
        if not result.params:
            print_optimization_result(
                output_line=output_line,
                parameter_lines=[],
                no_feasible_message="No feasible parameter set found (likely due to zero generated trades).",
            )
        else:
            parameter_lines = [f"roc_lookback={int(result.params['roc_lookback'])}"]
            if args.side == "call":
                parameter_lines.append(f"call_roc_threshold={result.params['call_roc_threshold']:.6f}")
                parameter_lines.append(f"upside_vol_threshold={result.params['upside_vol_threshold_annualized']:.6f}")
            else:
                parameter_lines.append(f"put_roc_threshold={result.params['put_roc_threshold']:.6f}")
                parameter_lines.append(f"downside_vol_threshold={result.params['downside_vol_threshold_annualized']:.6f}")
            parameter_lines.append(f"vol_window={int(result.params['vol_window'])}")
            parameter_lines.append("expiry=this-week")
            parameter_lines.append(f"objective={args.opt_goal_function}")
            parameter_lines.append(f"objective_score={result.score:.6f}")
            best_cmd = _format_best_backtest_command(
                script_path=script_name,
                symbol=args.symbol,
                side=args.side,
                params=result.params,
                csv_path=args.csv,
                start_date=args.start_date,
                end_date=args.end_date,
                allow_overlap=args.allow_overlap,
            )
            parameter_lines.append(f"cmd={best_cmd}")
            print_optimization_result(output_line=output_line, parameter_lines=parameter_lines)
            replay_flags = _format_best_backtest_flags(
                symbol=args.symbol,
                side=args.side,
                params=result.params,
                csv_path=args.csv,
                start_date=args.start_date,
                end_date=args.end_date,
                allow_overlap=args.allow_overlap,
            )
    else:
        trades_df = run_backtest(
            df=df,
            side=args.side,
            roc_lookback=args.roc_lookback,
            put_roc_threshold=args.put_roc_threshold,
            call_roc_threshold=args.call_roc_threshold,
            vol_window=args.vol_window,
            downside_vol_threshold_annualized=args.downside_vol_threshold,
            upside_vol_threshold_annualized=args.upside_vol_threshold,
            risk_free_rate=args.risk_free_rate,
            min_pricing_vol_annualized=args.min_pricing_vol,
            contract_size=args.contract_size,
            allow_overlap=args.allow_overlap,
        )

    print_summary_with_output_line(trades_df, output_line)

    show_all_trades = args.optimize
    print_recent_trades(
        trades_df=trades_df,
        columns=[
            "side",
            "entry_date",
            "exit_date",
            "scheduled_expiry_date",
            "entry_close",
            "exit_close",
            "premium",
            "intrinsic_at_expiry",
            "expired_itm",
            "pnl_per_contract",
            "time_to_expiry_days",
            "roc_signal",
            "trend_vol_signal",
        ],
        output_line=output_line,
        commented_output=commented_output,
        show_all_trades=show_all_trades,
        print_trades=args.print_trades,
    )

    if replay_flags:
        print(replay_flags)

    if args.trades_out:
        trades_df.to_csv(args.trades_out, index=False)
        print(f"\nSaved trades to: {args.trades_out}")


if __name__ == "__main__":
    main()
