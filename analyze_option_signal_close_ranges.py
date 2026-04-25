#!/usr/bin/env python3
"""Compute close-price ranges that trigger each option signal config row.

For each config row, this script appends a hypothetical evaluation-day bar with a
candidate close and reuses option_signal_notifier._evaluate_config to decide whether
that row would fire. It then reports interval(s) of close values where the alert is true.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from option_signal_notifier import SignalConfig, _evaluate_config, _load_configs, load_symbol_data


def _default_eval_date(history_end: pd.Timestamp) -> pd.Timestamp:
    # Use next business day by default so Friday guard in entry logic does not force zero signals.
    return (history_end + pd.offsets.BDay(1)).normalize()


def _append_hypothetical_bar(base_df: pd.DataFrame, eval_date: pd.Timestamp, close: float) -> pd.DataFrame:
    df = base_df.copy()
    new_row = {col: (df[col].iloc[-1] if col in df.columns else None) for col in df.columns}
    new_row["date"] = eval_date
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            new_row[col] = float(close)
    if "volume" in df.columns:
        last_volume = df["volume"].iloc[-1]
        new_row["volume"] = float(last_volume) if pd.notna(last_volume) else 0.0
    out = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def _fired_for_close(
    cfg: SignalConfig,
    base_df: pd.DataFrame,
    eval_date: pd.Timestamp,
    close: float,
    risk_free_rate: float,
    min_pricing_vol: float,
    contract_size: int,
) -> bool:
    hypo_df = _append_hypothetical_bar(base_df, eval_date, close)
    result = _evaluate_config(
        cfg=cfg,
        symbol_df=hypo_df,
        risk_free_rate=risk_free_rate,
        min_pricing_vol=min_pricing_vol,
        contract_size=contract_size,
    )
    return bool(result["fired_signals"])


def _find_true_intervals(
    predicate,
    lo: float,
    hi: float,
    grid_points: int,
    refine_steps: int,
) -> List[Dict[str, Any]]:
    if not (lo < hi):
        raise ValueError(f"Invalid close range: lo={lo}, hi={hi}")
    if grid_points < 2:
        raise ValueError("grid_points must be >= 2")

    xs = np.linspace(lo, hi, grid_points)
    vals = [predicate(float(x)) for x in xs]

    intervals: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(vals):
        if v and start is None:
            start = i
        if not v and start is not None:
            intervals.append((start, i - 1))
            start = None
    if start is not None:
        intervals.append((start, len(vals) - 1))

    out: List[Dict[str, Any]] = []

    def lower_boundary(a: float, b: float) -> float:
        # a=False, b=True
        for _ in range(refine_steps):
            m = (a + b) / 2.0
            if predicate(m):
                b = m
            else:
                a = m
        return b

    def upper_boundary(a: float, b: float) -> float:
        # a=True, b=False
        for _ in range(refine_steps):
            m = (a + b) / 2.0
            if predicate(m):
                a = m
            else:
                b = m
        return a

    for s, e in intervals:
        left_edge = s == 0
        right_edge = e == len(xs) - 1

        if left_edge:
            left = lo
        else:
            left = lower_boundary(float(xs[s - 1]), float(xs[s]))

        if right_edge:
            right = hi
        else:
            right = upper_boundary(float(xs[e]), float(xs[e + 1]))

        out.append(
            {
                "close_min": float(left),
                "close_max": float(right),
                "touches_search_min": bool(left_edge),
                "touches_search_max": bool(right_edge),
            }
        )

    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Find close-price ranges that trigger each option signal alert.")
    p.add_argument("--config", default="option_signal_notifier.config", help="Notifier config file.")
    p.add_argument("--csv", default=None, help="Optional data CSV. If omitted, notifier data loading is used.")
    p.add_argument("--start-date", default=None, help="Optional history start filter YYYY-MM-DD.")
    p.add_argument("--end-date", default=None, help="Optional history end filter YYYY-MM-DD.")
    p.add_argument("--eval-date", default=None, help="Hypothetical bar date YYYY-MM-DD. Default is next business day after history end.")
    p.add_argument("--close-min", type=float, default=None, help="Absolute minimum close for search.")
    p.add_argument("--close-max", type=float, default=None, help="Absolute maximum close for search.")
    p.add_argument("--close-min-factor", type=float, default=0.25, help="Fallback min = last_close * factor.")
    p.add_argument("--close-max-factor", type=float, default=2.50, help="Fallback max = last_close * factor.")
    p.add_argument("--grid-points", type=int, default=1200, help="Grid points for coarse scan.")
    p.add_argument("--refine-steps", type=int, default=20, help="Bisection steps for each boundary.")
    p.add_argument("--risk-free-rate", type=float, default=0.04)
    p.add_argument("--min-pricing-vol", type=float, default=0.10)
    p.add_argument("--contract-size", type=int, default=100)
    p.add_argument("--output-json", default=None, help="Optional JSON output path.")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg_args = argparse.Namespace(
        config=args.config,
        symbol="SPY",
        side="both",
        signal_model="roc",
        roc_lookback=5,
        accel_window=5,
        vol_window=20,
        put_roc_threshold=-0.03,
        call_roc_threshold=0.03,
        put_accel_threshold=-0.03,
        call_accel_threshold=0.03,
        downside_vol_threshold=0.20,
        upside_vol_threshold=0.20,
    )
    configs = _load_configs(cfg_args)

    symbol_data: Dict[str, pd.DataFrame] = {}
    for symbol in sorted({c.symbol for c in configs}):
        df = load_symbol_data(symbol, args.start_date, args.end_date, args.csv).copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df.empty:
            raise ValueError(f"No data for symbol {symbol}")
        symbol_data[symbol] = df.reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for i, cfg in enumerate(configs, start=1):
        base_df = symbol_data[cfg.symbol]
        history_end = pd.Timestamp(base_df["date"].max()).normalize()
        eval_date = pd.to_datetime(args.eval_date).normalize() if args.eval_date else _default_eval_date(history_end)

        # Avoid duplicate-date ambiguity by requiring eval date after history end.
        if eval_date <= history_end:
            raise ValueError(
                f"Config #{i} ({cfg.symbol}): eval-date {eval_date.date()} must be after history end {history_end.date()}"
            )

        last_close = float(base_df["close"].iloc[-1])
        lo = args.close_min if args.close_min is not None else max(0.01, last_close * args.close_min_factor)
        hi = args.close_max if args.close_max is not None else max(lo + 1e-9, last_close * args.close_max_factor)

        @lru_cache(maxsize=None)
        def pred_cached(x_rounded: float) -> bool:
            return _fired_for_close(
                cfg=cfg,
                base_df=base_df,
                eval_date=eval_date,
                close=float(x_rounded),
                risk_free_rate=args.risk_free_rate,
                min_pricing_vol=args.min_pricing_vol,
                contract_size=args.contract_size,
            )

        def pred(x: float) -> bool:
            # Float normalization keeps cache stable during bisection.
            return pred_cached(round(float(x), 10))

        intervals = _find_true_intervals(
            predicate=pred,
            lo=float(lo),
            hi=float(hi),
            grid_points=args.grid_points,
            refine_steps=args.refine_steps,
        )

        closest_pct = None
        if intervals:
            dists = []
            for iv in intervals:
                if iv["close_min"] <= last_close <= iv["close_max"]:
                    dists.append(0.0)
                elif last_close < iv["close_min"]:
                    dists.append((iv["close_min"] - last_close) / last_close)
                else:
                    dists.append((last_close - iv["close_max"]) / last_close)
            closest_pct = float(min(dists))

        rows.append(
            {
                "config_index": i,
                "symbol": cfg.symbol,
                "side": cfg.side,
                "signal_model": cfg.signal_model,
                "history_end": history_end.date().isoformat(),
                "eval_date": eval_date.date().isoformat(),
                "last_close": last_close,
                "search_close_min": float(lo),
                "search_close_max": float(hi),
                "interval_count": len(intervals),
                "closest_interval_distance_pct_from_last_close": closest_pct,
                "intervals": intervals,
                "config": asdict(cfg),
            }
        )

    payload = {
        "config_count": len(rows),
        "close_range_search": {
            "close_min": args.close_min,
            "close_max": args.close_max,
            "close_min_factor": args.close_min_factor,
            "close_max_factor": args.close_max_factor,
            "grid_points": args.grid_points,
            "refine_steps": args.refine_steps,
        },
        "results": rows,
    }

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote close-trigger ranges to {args.output_json}")

    for row in rows:
        print(
            f"#{row['config_index']:02d} {row['symbol']} {row['side']} {row['signal_model']} | "
            f"last_close={row['last_close']:.4f} | eval_date={row['eval_date']} | intervals={row['interval_count']}"
        )
        if not row["intervals"]:
            print("  - no trigger interval found in search range")
            continue
        for iv in row["intervals"]:
            suffix = ""
            if iv["touches_search_min"] or iv["touches_search_max"]:
                suffix = " (hits search boundary; widen range to confirm full interval)"
            print(f"  - [{iv['close_min']:.6f}, {iv['close_max']:.6f}]{suffix}")


if __name__ == "__main__":
    main()
