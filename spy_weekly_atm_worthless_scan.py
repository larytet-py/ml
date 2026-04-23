#!/usr/bin/env python3
"""
Scan weekly 0-5 DTE ATM short option outcomes from OHLCV data.

Strategy model:
- Entry day: any trading day in the week (Mon-Fri in data).
- Expiry day: any trading day from entry day through that week's final trading day.
- Strikes from entry-day open:
  - Call strike = ceil(open)   [rounded up/high]
  - Put strike  = floor(open)  [rounded down/low]
- "Worthless" at expiry:
  - Call worthless if expiry close <= call_strike
  - Put worthless if expiry close >= put_strike
  - Both worthless if both conditions hold

Example:
python3 spy_weekly_atm_worthless_scan.py \
  --csv data/etfs-all.csv \
  --symbol SPY \
  --output-csv data/spy_weekly_atm_worthless_scan.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import pandas as pd


def call_strike_from_open(entry_open: float) -> int:
    """ATM call strike rounded up (high side)."""
    return int(math.ceil(entry_open))


def put_strike_from_open(entry_open: float) -> int:
    """ATM put strike rounded down (low side)."""
    return int(math.floor(entry_open))


def load_ohlcv(csv_path: str, symbol: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "symbol" not in df.columns:
        raise ValueError("CSV must include a 'symbol' column.")
    if "date" not in df.columns:
        raise ValueError("CSV must include a 'date' column.")

    need_cols = ["open", "close", "high", "low", "volume"]
    for col in need_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must include a '{col}' column.")

    df = df[df["symbol"].astype(str).str.upper() == symbol.upper()].copy()
    if df.empty:
        raise ValueError(f"No rows found for symbol '{symbol}' in {csv_path}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in need_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "open", "close", "high", "low"])

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows left after filtering by symbol/date.")

    # Keep standard weekday sessions only (Mon..Fri), as requested.
    df = df[df["date"].dt.weekday <= 4].copy()
    if df.empty:
        raise ValueError("No Mon-Fri rows left after filtering.")

    return df.reset_index(drop=True)


def classify_price_behavior(
    period_high: float,
    period_low: float,
    expiry_close: float,
    reference_open: float,
    narrow_range_threshold_pct: float,
    top_zone_pct: float,
    low_zone_pct: float,
) -> str:
    width = max(0.0, period_high - period_low)
    if reference_open <= 0:
        return "unknown"

    range_pct = width / reference_open
    if range_pct <= narrow_range_threshold_pct:
        return "consolidating_narrow"

    if width == 0.0:
        return "consolidating_narrow"

    pos = (expiry_close - period_low) / width
    if pos >= top_zone_pct:
        return "constrained_top"
    if pos <= low_zone_pct:
        return "constrained_low"
    return "mid_range"


def scan_weekly_atm_worthless(
    df: pd.DataFrame,
    narrow_range_threshold_pct: float,
    top_zone_pct: float,
    low_zone_pct: float,
) -> pd.DataFrame:
    iso = df["date"].dt.isocalendar()
    df = df.copy()
    df["week_key"] = iso["year"].astype(int) * 100 + iso["week"].astype(int)

    out_rows = []

    for _, week in df.groupby("week_key", sort=True):
        week = week.sort_values("date").reset_index(drop=True)
        if week.empty:
            continue

        last_idx = len(week) - 1

        for i in range(len(week)):
            entry = week.iloc[i]
            entry_date = pd.Timestamp(entry["date"])
            entry_open = float(entry["open"])
            call_strike = call_strike_from_open(entry_open)
            put_strike = put_strike_from_open(entry_open)

            for j in range(i, last_idx + 1):
                expiry = week.iloc[j]
                expiry_date = pd.Timestamp(expiry["date"])
                expiry_close = float(expiry["close"])

                dte_calendar = int((expiry_date - entry_date).days)
                dte_trading = j - i

                if dte_calendar < 0 or dte_calendar > 5:
                    continue

                period = week.iloc[i : j + 1]
                period_high = float(period["high"].max())
                period_low = float(period["low"].min())
                period_range = period_high - period_low
                period_range_pct_of_open = (period_range / entry_open) if entry_open > 0 else float("nan")

                call_worthless = expiry_close <= call_strike
                put_worthless = expiry_close >= put_strike
                both_worthless = call_worthless and put_worthless

                if both_worthless:
                    outcome = "both"
                elif call_worthless and not put_worthless:
                    outcome = "call"
                elif put_worthless and not call_worthless:
                    outcome = "put"
                else:
                    outcome = "none"

                behavior = classify_price_behavior(
                    period_high=period_high,
                    period_low=period_low,
                    expiry_close=expiry_close,
                    reference_open=entry_open,
                    narrow_range_threshold_pct=narrow_range_threshold_pct,
                    top_zone_pct=top_zone_pct,
                    low_zone_pct=low_zone_pct,
                )

                out_rows.append(
                    {
                        "symbol": entry["symbol"],
                        "entry_date": entry_date.date().isoformat(),
                        "expiry_date": expiry_date.date().isoformat(),
                        "dte_calendar": dte_calendar,
                        "dte_trading": dte_trading,
                        "entry_open": round(entry_open, 4),
                        "call_strike_up": call_strike,
                        "put_strike_down": put_strike,
                        "expiry_close": round(expiry_close, 4),
                        "call_worthless": bool(call_worthless),
                        "put_worthless": bool(put_worthless),
                        "both_worthless": bool(both_worthless),
                        "outcome": outcome,
                        "period_high": round(period_high, 4),
                        "period_low": round(period_low, 4),
                        "period_range": round(period_range, 4),
                        "period_range_pct_of_open": round(period_range_pct_of_open, 6),
                        "price_behavior": behavior,
                    }
                )

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out

    out = out.sort_values(["entry_date", "expiry_date"]).reset_index(drop=True)
    return out


def print_summary(result_df: pd.DataFrame) -> None:
    if result_df.empty:
        print("No candidate trades found.")
        return

    total = len(result_df)
    call_wins = int(result_df["call_worthless"].sum())
    put_wins = int(result_df["put_worthless"].sum())
    both_wins = int(result_df["both_worthless"].sum())

    print(f"Rows scanned: {total}")
    print(f"Call worthless: {call_wins} ({call_wins / total:.2%})")
    print(f"Put worthless:  {put_wins} ({put_wins / total:.2%})")
    print(f"Both worthless: {both_wins} ({both_wins / total:.2%})")

    print("\nOutcome counts:")
    print(result_df["outcome"].value_counts().to_string())

    print("\nPrice behavior counts:")
    print(result_df["price_behavior"].value_counts().to_string())

    print("\nOutcome x Price behavior:")
    ctab = pd.crosstab(result_df["outcome"], result_df["price_behavior"])
    print(ctab.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan SPY ATM weekly 0-5 DTE worthless expiries.")
    parser.add_argument("--csv", default="data/etfs-all.csv", help="Input OHLCV CSV path")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol")
    parser.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--narrow-range-threshold-pct",
        type=float,
        default=0.015,
        help="Range <= this fraction of entry open is labeled consolidating_narrow (default 0.015 = 1.5%%)",
    )
    parser.add_argument(
        "--top-zone-pct",
        type=float,
        default=0.8,
        help="Expiry position in period range >= this is constrained_top",
    )
    parser.add_argument(
        "--low-zone-pct",
        type=float,
        default=0.2,
        help="Expiry position in period range <= this is constrained_low",
    )
    parser.add_argument(
        "--output-csv",
        default="data/spy_weekly_atm_worthless_scan.csv",
        help="Path to save detailed rows",
    )
    parser.add_argument(
        "--worthless-only-output-csv",
        default="data/spy_weekly_atm_worthless_only.csv",
        help="Path to save only rows where put/call/both are worthless",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.low_zone_pct <= 1.0 and 0.0 <= args.top_zone_pct <= 1.0):
        raise ValueError("--low-zone-pct and --top-zone-pct must be in [0,1].")

    df = load_ohlcv(
        csv_path=args.csv,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    result_df = scan_weekly_atm_worthless(
        df=df,
        narrow_range_threshold_pct=args.narrow_range_threshold_pct,
        top_zone_pct=args.top_zone_pct,
        low_zone_pct=args.low_zone_pct,
    )

    print_summary(result_df)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved detailed output: {output_path}")

    worthless_df = result_df[result_df["outcome"].isin(["call", "put", "both"])].copy()
    worthless_output_path = Path(args.worthless_only_output_csv)
    worthless_output_path.parent.mkdir(parents=True, exist_ok=True)
    worthless_df.to_csv(worthless_output_path, index=False)
    print(f"Saved worthless-only output: {worthless_output_path}")


if __name__ == "__main__":
    main()
