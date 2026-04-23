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


def load_ohlcv_for_symbol(
    source_df: pd.DataFrame,
    symbol: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    df = source_df.copy()

    if "symbol" not in df.columns:
        raise ValueError("CSV must include a 'symbol' column.")
    if "date" not in df.columns:
        raise ValueError("CSV must include a 'date' column.")

    need_cols = ["open", "close", "high", "low", "volume"]
    for col in need_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must include a '{col}' column.")

    df = df[df["symbol"].astype(str) == symbol].copy()
    if df.empty:
        raise ValueError(f"No rows found for symbol '{symbol}'.")

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


def load_source_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = ["symbol", "date", "open", "close", "high", "low", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        missing_str = ", ".join(f"'{c}'" for c in missing)
        raise ValueError(f"CSV must include columns: {missing_str}")

    return df


def infer_symbols(source_df: pd.DataFrame) -> list[str]:
    invalid_values = {"", "nan", "none", "null"}
    symbols = (
        source_df["symbol"]
        .astype(str)
        .str.strip()
        .loc[lambda s: ~s.str.lower().isin(invalid_values)]
        .dropna()
        .unique()
        .tolist()
    )
    symbols = sorted(symbols)
    if not symbols:
        raise ValueError("No symbols found in CSV 'symbol' column.")
    return symbols


def scan_weekly_atm_worthless(
    df: pd.DataFrame,
) -> pd.DataFrame:
    iso = df["date"].dt.isocalendar()
    df = df.copy()
    df["week_key"] = iso["year"].astype(int) * 100 + iso["week"].astype(int)

    # If data ends mid-week (Mon-Thu), trailing rows do not represent actual weekly expiration.
    # Drop that trailing partial week to avoid false "expired" outcomes caused by dataset truncation.
    last_date = pd.Timestamp(df["date"].max())
    if int(last_date.weekday()) < 4:
        last_iso = last_date.isocalendar()
        last_week_key = int(last_iso.year) * 100 + int(last_iso.week)
        df = df[df["week_key"] != last_week_key].copy()
        if df.empty:
            return pd.DataFrame()

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

                if both_worthless:
                    regime = "consolidating"
                elif call_worthless and not put_worthless:
                    regime = "constraining_top"
                elif put_worthless and not call_worthless:
                    regime = "constraining_low"
                else:
                    regime = "unclassified"

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
                        "is_consolidation": bool(both_worthless),
                        "regime": regime,
                        "outcome": outcome,
                        "period_high": round(period_high, 4),
                        "period_low": round(period_low, 4),
                        "period_range": round(period_range, 4),
                        "period_range_pct_of_open": round(period_range_pct_of_open, 6),
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

    print("\nRegime counts:")
    print(result_df["regime"].value_counts().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan ATM weekly 0-5 DTE worthless expiries.")
    parser.add_argument("--csv", default="data/etfs.csv", help="Input OHLCV CSV path")
    parser.add_argument(
        "--symbol",
        "--ticker",
        dest="symbol",
        default=None,
        help="Ticker symbol (case-sensitive). If omitted, process all symbols in --csv.",
    )
    parser.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path to save detailed rows (default: data/weekly_atm_worthless_scan.csv)",
    )
    parser.add_argument(
        "--worthless-only-output-csv",
        default=None,
        help="Path to save only rows where put/call/both are worthless (default: data/_weekly_atm_worthless_only.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_df = load_source_csv(args.csv)

    symbols = [args.symbol] if args.symbol else infer_symbols(source_df)
    total_symbols = len(symbols)
    if total_symbols > 1:
        print(f"Processing {total_symbols} symbols from {args.csv}")

    all_results: list[pd.DataFrame] = []

    for idx, symbol in enumerate(symbols, start=1):
        try:
            df = load_ohlcv_for_symbol(
                source_df=source_df,
                symbol=symbol,
                start_date=args.start_date,
                end_date=args.end_date,
            )
        except ValueError as exc:
            print(f"\n[{idx}/{total_symbols}] {symbol}: skipped ({exc})")
            continue

        result_df = scan_weekly_atm_worthless(df=df)

        print(f"\n[{idx}/{total_symbols}] Symbol: {symbol}")
        print_summary(result_df)

        if not result_df.empty:
            all_results.append(result_df)

    combined_df = (
        pd.concat(all_results, ignore_index=True).sort_values(["symbol", "entry_date", "expiry_date"]).reset_index(drop=True)
        if all_results
        else pd.DataFrame()
    )
    print("\nCombined summary:")
    print_summary(combined_df)

    output_csv = args.output_csv or "data/weekly_atm_worthless_scan.csv"
    worthless_only_output_csv = args.worthless_only_output_csv or "data/_weekly_atm_worthless_only.csv"

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    print(f"Saved detailed output: {output_path}")

    worthless_df = combined_df[combined_df["outcome"].isin(["call", "put", "both"])].copy()
    worthless_output_path = Path(worthless_only_output_csv)
    worthless_output_path.parent.mkdir(parents=True, exist_ok=True)
    worthless_df.to_csv(worthless_output_path, index=False)
    print(f"Saved worthless-only output: {worthless_output_path}")




if __name__ == "__main__":
    main()
