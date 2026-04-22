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
  --optimize \
  --opt-iters 50 \
  --opt-min-trades 5

python3 backtest_weekly_option_reversal.py \
  --symbol SPY \
  --side call \
  --optimize \
  --opt-iters 50 \
  --opt-min-trades 5
"""
import argparse
import math
import random
import shlex
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from option_pricing import black_scholes_call_price, black_scholes_put_price

TRADING_DAYS_PER_YEAR = 252
PRICING_VOL_WINDOW_DAYS = 21


def _shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


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
        "--print-trades",
        "-1",
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
    return _shell_join(cmd)


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


@dataclass
class OptimizationResult:
    params: Dict[str, float]
    score: float
    trades_df: pd.DataFrame
    metrics: Optional[Dict[str, float]]


def downside_std(x) -> float:
    negatives = x[x < 0]
    if len(negatives) < 2:
        return 0.0
    return float(negatives.std(ddof=0))


def upside_std(x) -> float:
    positives = x[x > 0]
    if len(positives) < 2:
        return 0.0
    return float(positives.std(ddof=0))


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
) -> pd.DataFrame:
    close = df["close"]
    returns = close.pct_change()

    df["roc"] = close / close.shift(roc_lookback) - 1.0
    df["downside_vol_annualized"] = (
        returns.rolling(vol_window).apply(downside_std, raw=True) * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    df["upside_vol_annualized"] = (
        returns.rolling(vol_window).apply(upside_std, raw=True) * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    # Keep Black-Scholes sigma on a fixed 21-trading-day realized-vol estimate.
    df["pricing_vol_annualized"] = (
        returns.rolling(PRICING_VOL_WINDOW_DAYS).std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)
    )

    trades = []
    next_entry_idx = 0

    # Use the last trading row within the same ISO week as expiry.
    iso = df["date"].dt.isocalendar()
    week_key = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    week_last_idx = pd.Series(df.index, index=df.index).groupby(week_key).transform("max").astype(int)

    for i in range(len(df)):
        if i < next_entry_idx:
            continue

        row = df.iloc[i]
        if pd.isna(row["roc"]) or pd.isna(row["pricing_vol_annualized"]):
            continue

        if side == "put":
            if pd.isna(row["downside_vol_annualized"]):
                continue
            trigger = (
                row["roc"] <= put_roc_threshold
                and row["downside_vol_annualized"] >= downside_vol_threshold_annualized
            )
            trend_vol_signal = float(row["downside_vol_annualized"])
        else:
            if pd.isna(row["upside_vol_annualized"]):
                continue
            trigger = (
                row["roc"] >= call_roc_threshold
                and row["upside_vol_annualized"] >= upside_vol_threshold_annualized
            )
            trend_vol_signal = float(row["upside_vol_annualized"])

        if not trigger:
            continue

        entry_close = float(row["close"])
        entry_date = pd.Timestamp(row["date"])
        strike = entry_close
        exit_idx = int(week_last_idx.iloc[i])
        days_to_friday = 4 - entry_date.weekday()
        if days_to_friday <= 0:
            continue
        scheduled_expiry_date = (entry_date + pd.Timedelta(days=days_to_friday)).normalize()
        time_to_expiry_days = days_to_friday
        time_to_expiry_years = time_to_expiry_days / 365.25

        if exit_idx >= len(df):
            continue
        if exit_idx <= i:
            # Skip same-day expiry signals (e.g., Friday close) in this EOD model.
            continue

        pricing_vol = max(float(row["pricing_vol_annualized"]), min_pricing_vol_annualized)
        if side == "put":
            premium = black_scholes_put_price(
                spot=entry_close,
                strike=strike,
                time_to_expiry_years=time_to_expiry_years,
                risk_free_rate=risk_free_rate,
                sigma=pricing_vol,
            )
        else:
            premium = black_scholes_call_price(
                spot=entry_close,
                strike=strike,
                time_to_expiry_years=time_to_expiry_years,
                risk_free_rate=risk_free_rate,
                sigma=pricing_vol,
            )

        exit_row = df.iloc[exit_idx]
        exit_close = float(exit_row["close"])
        intrinsic = max(strike - exit_close, 0.0) if side == "put" else max(exit_close - strike, 0.0)
        expired_itm = intrinsic > 0.0
        pnl_per_share = premium - intrinsic

        trades.append(
            Trade(
                side=side,
                entry_date=row["date"],
                exit_date=exit_row["date"],
                entry_close=entry_close,
                exit_close=exit_close,
                strike=strike,
                premium=premium,
                intrinsic_at_expiry=intrinsic,
                expired_itm=expired_itm,
                pnl_per_share=pnl_per_share,
                pnl_per_contract=pnl_per_share * contract_size,
                roc_signal=float(row["roc"]),
                trend_vol_signal=trend_vol_signal,
                pricing_vol=pricing_vol,
                scheduled_expiry_date=scheduled_expiry_date,
                time_to_expiry_days=time_to_expiry_days,
            )
        )

        if not allow_overlap:
            next_entry_idx = exit_idx + 1

    return pd.DataFrame([t.__dict__ for t in trades])


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
    progress_interval_seconds: float,
) -> OptimizationResult:
    if any(lo >= hi for lo, hi in bounds):
        raise ValueError("Invalid optimization bounds: each min value must be less than max.")

    rng = random.Random(seed)
    vector_len = len(initial_vector)
    int_dims = {0, 2}

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
        trades_df = run_backtest(
            df=df,
            side=side,
            roc_lookback=int(params["roc_lookback"]),
            put_roc_threshold=float(params["put_roc_threshold"]),
            call_roc_threshold=float(params["call_roc_threshold"]),
            vol_window=int(params["vol_window"]),
            downside_vol_threshold_annualized=float(params["downside_vol_threshold_annualized"]),
            upside_vol_threshold_annualized=float(params["upside_vol_threshold_annualized"]),
            risk_free_rate=risk_free_rate,
            min_pricing_vol_annualized=min_pricing_vol_annualized,
            contract_size=contract_size,
            allow_overlap=allow_overlap,
        )
        metrics = summarize_trades(trades_df)
        if metrics is None:
            return -1_000_000_000.0, trades_df, metrics, params
        total_trades = int(metrics["total"])
        shortfall = max(0, min_trades - total_trades)
        if shortfall > 0:
            # Hard feasibility gate: do not let sparse, high-PnL outliers win optimization.
            # Keep a tiny tie-breaker so if no feasible point exists we still prefer more trades.
            infeasible_score = (
                -1_000_000.0
                - trade_penalty * float(shortfall)
                + 0.01 * float(total_trades)
                + 0.001 * float(metrics["avg_pnl"])
            )
            return infeasible_score, trades_df, metrics, params
        score = float(metrics["avg_pnl"])
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

    best_score = -1_000_000_000.0
    best_df = pd.DataFrame()
    best_metrics: Optional[Dict[str, float]] = None
    best_params: Dict[str, float] = {}
    eval_count = 0
    start_time = time.monotonic()
    last_report_time = start_time

    starts: List[List[float]] = [project(initial_vector)]
    for _ in range(max(0, restarts - 1)):
        starts.append([rng.uniform(lo, hi) for lo, hi in bounds])

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
                f"best_avg_pnl={best_metrics['avg_pnl']:.2f} "
                f"trades={int(best_metrics['total'])} "
                f"cmd={best_cmd}",
                flush=True,
            )
        last_report_time = now

    for start in starts:
        x = start.copy()
        score, trades_df, metrics, params = evaluate(x)
        eval_count += 1
        if score > best_score:
            best_score, best_df, best_metrics, best_params = score, trades_df, metrics, params
        maybe_report_progress()

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
                if candidate_score >= score:
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

            if score > best_score:
                best_score, best_df, best_metrics, best_params = score, trades_df, metrics, params
            maybe_report_progress()

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
    parser.add_argument("--opt-iters", type=int, default=50, help="Iterations per optimization restart.")
    parser.add_argument("--opt-restarts", type=int, default=20, help="Number of optimization restarts.")
    parser.add_argument("--opt-learning-rate", type=float, default=0.25, help="Projected gradient ascent step size.")
    parser.add_argument("--opt-fd-eps", type=float, default=0.03, help="Finite-difference relative step for continuous params.")
    parser.add_argument(
        "--opt-min-trades",
        type=int,
        default=5,
        help="Minimum trades target used in objective penalty.",
    )
    parser.add_argument(
        "--opt-trade-penalty",
        type=float,
        default=75.0,
        help="Penalty points per missing trade below --opt-min-trades.",
    )
    parser.add_argument("--opt-seed", type=int, default=7, help="Random seed for optimizer restarts.")
    parser.add_argument(
        "--opt-progress-seconds",
        type=float,
        default=1.0,
        help="How often to print optimization progress. Set <=0 to disable.",
    )
    parser.add_argument("--roc-lookback-min", type=int, default=2, help="Lower bound for roc-lookback in optimization.")
    parser.add_argument("--roc-lookback-max", type=int, default=40, help="Upper bound for roc-lookback in optimization.")
    parser.add_argument("--vol-window-min", type=int, default=5, help="Lower bound for vol-window in optimization.")
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
        default=0.01,
        help="Lower bound for side-specific vol threshold in optimization.",
    )
    parser.add_argument(
        "--vol-threshold-max",
        type=float,
        default=0.30,
        help="Upper bound for side-specific vol threshold in optimization.",
    )
    args = parser.parse_args()

    df = load_symbol_data(args.csv, args.symbol, args.start_date, args.end_date)

    if args.optimize:
        if args.side == "call":
            roc_threshold_min = args.roc_threshold_min if args.roc_threshold_min is not None else 0.005
            roc_threshold_max = args.roc_threshold_max if args.roc_threshold_max is not None else 0.10
            initial_roc_threshold = args.call_roc_threshold
            initial_vol_threshold = args.upside_vol_threshold
        else:
            roc_threshold_min = args.roc_threshold_min if args.roc_threshold_min is not None else -0.10
            roc_threshold_max = args.roc_threshold_max if args.roc_threshold_max is not None else -0.005
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
            script_path="backtest_weekly_option_reversal.py",
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
            progress_interval_seconds=args.opt_progress_seconds,
        )

        trades_df = result.trades_df
        print("Optimization complete. Best parameters:")
        print(f"  roc_lookback={int(result.params['roc_lookback'])}")
        if args.side == "call":
            print(f"  call_roc_threshold={result.params['call_roc_threshold']:.6f}")
            print(f"  upside_vol_threshold={result.params['upside_vol_threshold_annualized']:.6f}")
        else:
            print(f"  put_roc_threshold={result.params['put_roc_threshold']:.6f}")
            print(f"  downside_vol_threshold={result.params['downside_vol_threshold_annualized']:.6f}")
        print(f"  vol_window={int(result.params['vol_window'])}")
        print("  expiry=this-week")
        print(f"  objective_score={result.score:.2f}")
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

    print_summary(trades_df)

    if not trades_df.empty and args.print_trades != 0:
        print("\nRecent trades:")
        cols = [
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
        ]
        trades_to_show = trades_df[cols] if args.print_trades < 0 else trades_df[cols].tail(args.print_trades)
        print(trades_to_show.to_string(index=False, justify="center"))

    if args.trades_out:
        trades_df.to_csv(args.trades_out, index=False)
        print(f"\nSaved trades to: {args.trades_out}")


if __name__ == "__main__":
    main()
