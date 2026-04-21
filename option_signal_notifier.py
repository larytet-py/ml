#!/usr/bin/env python3
"""
Usage:

python3 option_signal_notifier.py \
  --symbol SPY \
  --side call \
  --roc-lookback 20 \
  --vol-window 17 \
  --call-roc-threshold 0.052641 \
  --upside-vol-threshold 0.085369

python3 option_signal_notifier.py \
  --symbol SPY \
  --side put \
  --roc-lookback 18 \
  --vol-window 20 \
  --put-roc-threshold -0.005000 \
  --downside-vol-threshold 0.094507
"""
import argparse
import io
import math
from typing import List, Optional

import pandas as pd
import requests

from option_pricing import black_scholes_call_price, black_scholes_put_price


TRADING_DAYS_PER_YEAR = 252


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


def load_symbol_data_from_csv(csv_path: str, symbol: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol].copy()
    if df.empty:
        raise ValueError(f"No rows found for symbol '{symbol}' in {csv_path}")

    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    for old_name, new_name in [("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"), ("Volume", "volume")]:
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "close", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    if "close" not in df.columns:
        raise ValueError("Input data must contain a 'close' column.")
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows left after date filters.")
    return df


def _fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    symbol_lower = symbol.lower()
    candidates = [f"{symbol_lower}.us", symbol_lower]
    last_error = None
    for candidate in candidates:
        url = f"https://stooq.com/q/d/l/?s={candidate}&i=d"
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            text = response.text.strip()
            if not text or "No data" in text:
                continue
            df = pd.read_csv(io.StringIO(text))
            if df.empty or "Close" not in df.columns:
                continue
            df = df.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            df["date"] = pd.to_datetime(df["date"])
            for col in ["open", "close", "high", "low", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
            return df
        except requests.RequestException as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise RuntimeError(f"Failed to download daily data for {symbol}: {last_error}")
    raise RuntimeError(f"No data returned for symbol {symbol} from stooq.")


def load_symbol_data(symbol: str, start_date: Optional[str], end_date: Optional[str], csv_path: Optional[str]) -> pd.DataFrame:
    if csv_path:
        return load_symbol_data_from_csv(csv_path, symbol, start_date, end_date)
    df = _fetch_stooq_daily(symbol)
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    if df.empty:
        raise ValueError("No rows left after date filters.")
    return df.reset_index(drop=True)


def build_signal_frame(df: pd.DataFrame, roc_lookback: int, vol_window: int) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    returns = close.pct_change()
    out["roc"] = close / close.shift(roc_lookback) - 1.0
    out["downside_vol_annualized"] = returns.rolling(vol_window).apply(downside_std, raw=True) * math.sqrt(TRADING_DAYS_PER_YEAR)
    out["upside_vol_annualized"] = returns.rolling(vol_window).apply(upside_std, raw=True) * math.sqrt(TRADING_DAYS_PER_YEAR)
    out["pricing_vol_annualized"] = returns.rolling(vol_window).std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)
    return out


def _days_to_next_friday(ts: pd.Timestamp) -> int:
    days = (4 - ts.weekday()) % 7
    return 7 if days == 0 else days


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily signal notifier: checks weekly reversal signal and prints put/call entry notification with estimated ATM premium."
    )
    parser.add_argument("--symbol", required=True, help="Ticker symbol, e.g. SPY")
    parser.add_argument("--side", choices=["put", "call", "both"], default="both", help="Which side(s) to evaluate.")
    parser.add_argument("--csv", default=None, help="Optional local CSV path; if omitted script downloads daily bars from stooq.")
    parser.add_argument("--start-date", default=None, help="Optional date filter, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Optional date filter, YYYY-MM-DD.")
    parser.add_argument("--roc-lookback", type=int, default=5, help="Days for ROC.")
    parser.add_argument("--vol-window", type=int, default=20, help="Rolling window for volatility.")
    parser.add_argument("--put-roc-threshold", type=float, default=-0.03, help="Put trigger: ROC <= threshold.")
    parser.add_argument("--call-roc-threshold", type=float, default=0.03, help="Call trigger: ROC >= threshold.")
    parser.add_argument("--downside-vol-threshold", type=float, default=0.20, help="Annualized downside volatility threshold.")
    parser.add_argument("--upside-vol-threshold", type=float, default=0.20, help="Annualized upside volatility threshold.")
    parser.add_argument("--risk-free-rate", type=float, default=0.04, help="Risk-free rate for Black-Scholes.")
    parser.add_argument("--min-pricing-vol", type=float, default=0.10, help="Vol floor (annualized) used in option pricing.")
    parser.add_argument("--contract-size", type=int, default=100, help="Shares per option contract.")
    parser.add_argument(
        "--print-trades",
        type=int,
        default=0,
        help="Print recent rows with computed signals. Use -1 for all rows, 0 to disable.",
    )
    args = parser.parse_args()

    df = load_symbol_data(args.symbol, args.start_date, args.end_date, args.csv)
    signal_df = build_signal_frame(df, args.roc_lookback, args.vol_window)
    latest = signal_df.iloc[-1]

    if pd.isna(latest["roc"]) or pd.isna(latest["pricing_vol_annualized"]):
        raise ValueError("Insufficient history to compute latest signal values with current lookback/window.")

    put_trigger = (
        bool(latest["roc"] <= args.put_roc_threshold)
        and not pd.isna(latest["downside_vol_annualized"])
        and bool(latest["downside_vol_annualized"] >= args.downside_vol_threshold)
    )
    call_trigger = (
        bool(latest["roc"] >= args.call_roc_threshold)
        and not pd.isna(latest["upside_vol_annualized"])
        and bool(latest["upside_vol_annualized"] >= args.upside_vol_threshold)
    )

    if args.side == "put":
        selected_sides: List[str] = ["put"]
    elif args.side == "call":
        selected_sides = ["call"]
    else:
        selected_sides = ["put", "call"]

    spot = float(latest["close"])
    latest_date = pd.Timestamp(latest["date"])
    days_to_expiry = _days_to_next_friday(latest_date)
    time_to_expiry_years = days_to_expiry / 365.25
    pricing_vol = max(float(latest["pricing_vol_annualized"]), args.min_pricing_vol)

    print(
        f"Latest bar: {latest_date.date()} | {args.symbol.upper()} close={spot:.2f} | "
        f"ROC({args.roc_lookback})={float(latest['roc']):.4%} | "
        f"downside_vol={float(latest['downside_vol_annualized']):.4%} | "
        f"upside_vol={float(latest['upside_vol_annualized']):.4%} | "
        f"pricing_vol={pricing_vol:.4%}"
    )

    messages = []
    for side in selected_sides:
        is_triggered = put_trigger if side == "put" else call_trigger
        if not is_triggered:
            continue
        if side == "put":
            premium = black_scholes_put_price(
                spot=spot,
                strike=spot,
                time_to_expiry_years=time_to_expiry_years,
                risk_free_rate=args.risk_free_rate,
                sigma=pricing_vol,
            )
        else:
            premium = black_scholes_call_price(
                spot=spot,
                strike=spot,
                time_to_expiry_years=time_to_expiry_years,
                risk_free_rate=args.risk_free_rate,
                sigma=pricing_vol,
            )
        messages.append(
            f"NOTIFICATION: ENTER {side.upper()} POSITION | expiry(next Friday in {days_to_expiry} days) | "
            f"estimated ATM premium={premium:.4f} per share ({premium * args.contract_size:.2f} per contract)"
        )

    if not messages:
        print("NOTIFICATION: No entry signal for selected side(s) on latest daily bar.")
    else:
        for msg in messages:
            print(msg)

    if args.print_trades != 0:
        printable = signal_df.copy()
        printable["put_signal"] = (
            (printable["roc"] <= args.put_roc_threshold)
            & (printable["downside_vol_annualized"] >= args.downside_vol_threshold)
        )
        printable["call_signal"] = (
            (printable["roc"] >= args.call_roc_threshold)
            & (printable["upside_vol_annualized"] >= args.upside_vol_threshold)
        )
        cols = [
            "date",
            "close",
            "roc",
            "downside_vol_annualized",
            "upside_vol_annualized",
            "pricing_vol_annualized",
            "put_signal",
            "call_signal",
        ]
        to_show = printable[cols] if args.print_trades < 0 else printable[cols].tail(args.print_trades)
        print("\nSignal history:")
        print(to_show.to_string(index=False, justify="center"))


if __name__ == "__main__":
    main()

