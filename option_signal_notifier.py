#!/usr/bin/env python3
"""
Usage:

MARKET_STACK_API_KEY=1fd.. python3 option_signal_notifier.py \
  --symbol SPY \
  --side call \
  --roc-lookback 20 \
  --vol-window 17 \
  --call-roc-threshold 0.052641 \
  --upside-vol-threshold 0.085369

MARKET_STACK_API_KEY=1fd.. python3 option_signal_notifier.py \
  --symbol SPY \
  --side put \
  --roc-lookback 18 \
  --vol-window 20 \
  --put-roc-threshold -0.005000 \
  --downside-vol-threshold 0.094507

MARKET_STACK_API_KEY=1fd.. python3 option_signal_notifier.py \
  --config option_signal_configs.txt

Example option_signal_configs.txt rows:
--symbol SPY --side put --roc-lookback 18 --vol-window 20 --put-roc-threshold -0.005000 --downside-vol-threshold 0.094507
--symbol SPY --side call --roc-lookback 20 --vol-window 17 --call-roc-threshold 0.052641 --upside-vol-threshold 0.085369
"""
import argparse
import json
import math
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from option_pricing import black_scholes_call_price, black_scholes_put_price


TRADING_DAYS_PER_YEAR = 252
PRICING_VOL_WINDOW_DAYS = 21
DEFAULT_CACHE_CSV = "data/etfs.csv"


@dataclass
class SignalConfig:
    symbol: str
    side: str
    roc_lookback: int
    vol_window: int
    put_roc_threshold: float
    call_roc_threshold: float
    downside_vol_threshold: float
    upside_vol_threshold: float


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
    # Filter out invalid placeholder rows (for example close=0) that break return/volatility math.
    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0]
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows left after date/price filters.")
    return df


def _fetch_marketstack_daily(symbol: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    api_key = os.getenv("MARKET_STACK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing MARKET_STACK_API_KEY environment variable.")

    base_url = "http://api.marketstack.com/v1/eod"
    limit = 1000
    offset = 0
    rows = []

    while True:
        params = {
            "access_key": api_key,
            "symbols": symbol.upper(),
            "limit": limit,
            "offset": offset,
            "sort": "ASC",
        }
        if start_date:
            params["date_from"] = start_date
        if end_date:
            params["date_to"] = end_date

        try:
            response = requests.get(base_url, params=params, timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to download daily data for {symbol} from marketstack: {exc}") from exc

        payload = response.json()
        if "error" in payload:
            message = payload["error"].get("message", "Unknown marketstack error")
            raise RuntimeError(f"marketstack API error for {symbol}: {message}")

        batch = payload.get("data", [])
        if not batch:
            break
        rows.extend(batch)

        pagination = payload.get("pagination", {})
        count = int(pagination.get("count", len(batch)))
        total = int(pagination.get("total", len(rows)))
        if count == 0 or offset + count >= total:
            break
        offset += count

    if not rows:
        raise RuntimeError(f"No data returned for symbol {symbol} from marketstack.")

    df = pd.DataFrame(rows)
    for col in ["open", "close", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "date" not in df.columns:
        raise RuntimeError(f"marketstack response for {symbol} did not include 'date'.")
    if "close" not in df.columns:
        raise RuntimeError(f"marketstack response for {symbol} did not include 'close'.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _is_recent_enough(latest_date: pd.Timestamp) -> bool:
    # Require cache to include the latest completed business day.
    # In practice this means only "today" can be missing.
    today = pd.Timestamp.today().normalize()
    latest_expected_trading_day = (today - pd.offsets.BDay(1)).normalize()
    return latest_date.normalize() >= latest_expected_trading_day


def _persist_symbol_rows(cache_csv_path: str, symbol: str, fresh_rows: pd.DataFrame) -> None:
    cache_path = Path(cache_csv_path)
    if cache_path.exists():
        full_df = pd.read_csv(cache_csv_path)
    else:
        full_df = pd.DataFrame(columns=["symbol", "date", "open", "close", "high", "low", "volume"])

    if "date" in full_df.columns:
        full_df["date"] = pd.to_datetime(full_df["date"], errors="coerce")

    if "symbol" not in full_df.columns:
        full_df["symbol"] = symbol.upper()

    keep_mask = full_df["symbol"].astype(str).str.upper() != symbol.upper()
    other_symbols = full_df[keep_mask].copy()

    symbol_df = fresh_rows.copy()
    symbol_df["symbol"] = symbol.upper()
    symbol_df["date"] = pd.to_datetime(symbol_df["date"], errors="coerce").dt.tz_localize(None)
    symbol_df = symbol_df.dropna(subset=["date", "close"])
    symbol_df = symbol_df[symbol_df["close"] > 0]
    symbol_df = symbol_df.drop_duplicates(subset=["date"], keep="last")
    symbol_df = symbol_df.sort_values("date")

    merged = pd.concat([other_symbols, symbol_df], ignore_index=True)
    merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_csv_path, index=False)


def _load_or_refresh_cached_data(
    symbol: str,
    start_date: Optional[str],
    end_date: Optional[str],
    cache_csv_path: str,
) -> Optional[pd.DataFrame]:
    cache_path = Path(cache_csv_path)
    if not cache_path.exists():
        return None

    try:
        cached = load_symbol_data_from_csv(cache_csv_path, symbol, None, None)
    except ValueError:
        return None

    if cached.empty:
        return None

    latest_cached_date = pd.Timestamp(cached["date"].max()).tz_localize(None)
    if _is_recent_enough(latest_cached_date):
        print(f"Using cached data from {cache_csv_path} (latest {latest_cached_date.date()}); skipping marketstack API.")
        return load_symbol_data_from_csv(cache_csv_path, symbol, start_date, end_date)

    api_key = os.getenv("MARKET_STACK_API_KEY")
    if not api_key:
        print(
            f"Cached data in {cache_csv_path} for {symbol.upper()} is stale (latest {latest_cached_date.date()}) and "
            "MARKET_STACK_API_KEY is not set; using cache without refresh."
        )
        return load_symbol_data_from_csv(cache_csv_path, symbol, start_date, end_date)

    missing_from = (latest_cached_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    target_to = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")

    try:
        fresh = _fetch_marketstack_daily(symbol, start_date=missing_from, end_date=target_to)
    except RuntimeError as exc:
        if "No data returned for symbol" in str(exc):
            print(
                f"Warning: cache refresh returned no rows for {symbol.upper()} in range "
                f"{missing_from}..{target_to}; using existing cached data."
            )
        else:
            print(f"Cache refresh failed ({exc}); using existing cached data.")
        return load_symbol_data_from_csv(cache_csv_path, symbol, start_date, end_date)

    if fresh.empty:
        print(
            f"Warning: cache refresh returned empty dataframe for {symbol.upper()} in range "
            f"{missing_from}..{target_to}; using existing cached data."
        )
        return load_symbol_data_from_csv(cache_csv_path, symbol, start_date, end_date)

    combined = pd.concat([cached, fresh], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.tz_localize(None)
    combined = combined.dropna(subset=["date", "close"])
    combined = combined[combined["close"] > 0]
    combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    _persist_symbol_rows(cache_csv_path, symbol, combined)

    updated_last = pd.Timestamp(combined["date"].max()).date()
    print(f"Updated {cache_csv_path} for {symbol.upper()} through {updated_last}.")
    return load_symbol_data_from_csv(cache_csv_path, symbol, start_date, end_date)


def load_symbol_data(symbol: str, start_date: Optional[str], end_date: Optional[str], csv_path: Optional[str]) -> pd.DataFrame:
    symbol = symbol.upper()
    if csv_path:
        return load_symbol_data_from_csv(csv_path, symbol, start_date, end_date)

    cached = _load_or_refresh_cached_data(symbol, start_date, end_date, DEFAULT_CACHE_CSV)
    if cached is not None and not cached.empty:
        return cached.reset_index(drop=True)

    df = _fetch_marketstack_daily(symbol, start_date, end_date)
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    if df.empty:
        raise ValueError("No rows left after date filters.")

    # Seed local cache for future runs when it is absent.
    try:
        _persist_symbol_rows(DEFAULT_CACHE_CSV, symbol, df)
        print(f"Saved {len(df)} rows to {DEFAULT_CACHE_CSV} for future runs.")
    except Exception as exc:  # pragma: no cover - non-fatal cache write path
        print(f"Warning: failed to persist cache to {DEFAULT_CACHE_CSV}: {exc}")

    return df.reset_index(drop=True)


def build_signal_frame(df: pd.DataFrame, roc_lookback: int, vol_window: int) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    returns = close.pct_change()
    out["roc"] = close / close.shift(roc_lookback) - 1.0
    out["downside_vol_annualized"] = returns.rolling(vol_window).apply(downside_std, raw=True) * math.sqrt(TRADING_DAYS_PER_YEAR)
    out["upside_vol_annualized"] = returns.rolling(vol_window).apply(upside_std, raw=True) * math.sqrt(TRADING_DAYS_PER_YEAR)
    # Keep pricing sigma on the same fixed 21-day realized-vol window as the backtest.
    out["pricing_vol_annualized"] = returns.rolling(PRICING_VOL_WINDOW_DAYS).std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)
    return out


def _entry_candidates_by_side(signal_df: pd.DataFrame, cfg: SignalConfig, side: str) -> Dict[int, int]:
    if side not in {"put", "call"}:
        raise ValueError(f"Unsupported side '{side}'. Expected 'put' or 'call'.")

    iso = signal_df["date"].dt.isocalendar()
    week_key = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    week_last_idx = pd.Series(signal_df.index, index=signal_df.index).groupby(week_key).transform("max").astype(int)

    entries: Dict[int, int] = {}
    next_entry_idx = 0

    for i in range(len(signal_df)):
        if i < next_entry_idx:
            continue

        row = signal_df.iloc[i]
        if pd.isna(row["roc"]) or pd.isna(row["pricing_vol_annualized"]):
            continue

        if side == "put":
            if pd.isna(row["downside_vol_annualized"]):
                continue
            is_triggered = bool(
                row["roc"] <= cfg.put_roc_threshold
                and row["downside_vol_annualized"] >= cfg.downside_vol_threshold
            )
        else:
            if pd.isna(row["upside_vol_annualized"]):
                continue
            is_triggered = bool(
                row["roc"] >= cfg.call_roc_threshold
                and row["upside_vol_annualized"] >= cfg.upside_vol_threshold
            )

        if not is_triggered:
            continue

        entry_date = pd.Timestamp(row["date"])
        days_to_friday = 4 - entry_date.weekday()
        if days_to_friday <= 0:
            # Match backtest EOD model: do not open same-day expiry positions.
            continue

        exit_idx = int(week_last_idx.iloc[i])
        if exit_idx >= len(signal_df) or exit_idx <= i:
            continue

        entries[i] = days_to_friday
        # Match backtest default behavior (allow_overlap=False).
        next_entry_idx = exit_idx + 1

    return entries


def _make_config_row_parser() -> argparse.ArgumentParser:
    row_parser = argparse.ArgumentParser(add_help=False)
    row_parser.add_argument("--symbol", required=True)
    row_parser.add_argument("--side", choices=["put", "call", "both"], default="both")
    row_parser.add_argument("--roc-lookback", type=int, default=5)
    row_parser.add_argument("--vol-window", type=int, default=20)
    row_parser.add_argument("--put-roc-threshold", type=float, default=-0.03)
    row_parser.add_argument("--call-roc-threshold", type=float, default=0.03)
    row_parser.add_argument("--downside-vol-threshold", type=float, default=0.20)
    row_parser.add_argument("--upside-vol-threshold", type=float, default=0.20)
    return row_parser


def _load_configs(args: argparse.Namespace) -> List[SignalConfig]:
    if not args.config:
        return [
            SignalConfig(
                symbol=args.symbol.upper(),
                side=args.side,
                roc_lookback=args.roc_lookback,
                vol_window=args.vol_window,
                put_roc_threshold=args.put_roc_threshold,
                call_roc_threshold=args.call_roc_threshold,
                downside_vol_threshold=args.downside_vol_threshold,
                upside_vol_threshold=args.upside_vol_threshold,
            )
        ]

    path = Path(args.config)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    row_parser = _make_config_row_parser()
    configs: List[SignalConfig] = []
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = shlex.split(line)
        try:
            row = row_parser.parse_args(parts)
        except SystemExit as exc:
            raise ValueError(f"Invalid config row at line {line_no}: {raw_line}") from exc

        configs.append(
            SignalConfig(
                symbol=row.symbol.upper(),
                side=row.side,
                roc_lookback=row.roc_lookback,
                vol_window=row.vol_window,
                put_roc_threshold=row.put_roc_threshold,
                call_roc_threshold=row.call_roc_threshold,
                downside_vol_threshold=row.downside_vol_threshold,
                upside_vol_threshold=row.upside_vol_threshold,
            )
        )

    if not configs:
        raise ValueError(f"No usable config rows in {args.config}")
    return configs


def _required_history(cfg: SignalConfig) -> int:
    return max(cfg.roc_lookback + 1, cfg.vol_window + 1)


def _evaluate_config(
    cfg: SignalConfig,
    symbol_df: pd.DataFrame,
    risk_free_rate: float,
    min_pricing_vol: float,
    contract_size: int,
) -> Dict[str, Any]:
    signal_df = build_signal_frame(symbol_df, cfg.roc_lookback, cfg.vol_window)
    latest = signal_df.iloc[-1]
    latest_idx = len(signal_df) - 1

    if pd.isna(latest["roc"]) or pd.isna(latest["pricing_vol_annualized"]):
        needed = _required_history(cfg)
        raise ValueError(
            f"Insufficient history for {cfg.symbol} {cfg.side} with lookback/window "
            f"({cfg.roc_lookback}/{cfg.vol_window}). Need at least {needed} rows, got {len(signal_df)} rows."
        )

    if cfg.side == "put":
        selected_sides: List[str] = ["put"]
    elif cfg.side == "call":
        selected_sides = ["call"]
    else:
        selected_sides = ["put", "call"]

    spot = float(latest["close"])
    latest_date = pd.Timestamp(latest["date"])
    latest_roc = float(latest["roc"])
    latest_downside_vol = float(latest["downside_vol_annualized"])
    latest_upside_vol = float(latest["upside_vol_annualized"])
    latest_pricing_vol = float(latest["pricing_vol_annualized"])
    pricing_vol = max(latest_pricing_vol, min_pricing_vol)

    entry_candidates: Dict[str, Dict[int, int]] = {}
    for side in selected_sides:
        entry_candidates[side] = _entry_candidates_by_side(signal_df, cfg, side)

    put_trigger = "put" in entry_candidates and latest_idx in entry_candidates["put"]
    call_trigger = "call" in entry_candidates and latest_idx in entry_candidates["call"]

    messages: List[str] = []
    fired_signals: List[Dict[str, Any]] = []
    fired = False
    for side in selected_sides:
        is_triggered = put_trigger if side == "put" else call_trigger
        if not is_triggered:
            continue
        days_to_expiry = int(entry_candidates[side][latest_idx])
        time_to_expiry_years = days_to_expiry / 365.25

        if side == "put":
            premium = black_scholes_put_price(
                spot=spot,
                strike=spot,
                time_to_expiry_years=time_to_expiry_years,
                risk_free_rate=risk_free_rate,
                sigma=pricing_vol,
            )
        else:
            premium = black_scholes_call_price(
                spot=spot,
                strike=spot,
                time_to_expiry_years=time_to_expiry_years,
                risk_free_rate=risk_free_rate,
                sigma=pricing_vol,
            )

        fired = True
        premium_per_contract = premium * contract_size
        signal_message = (
            f"ENTER {side.upper()} POSITION | {cfg.symbol} | next Friday in {days_to_expiry} days | "
            f"estimated ATM premium={premium:.4f} per share, last close {spot:.2f}"
        )
        messages.append(signal_message)
        fired_signals.append(
            {
                "side": side,
                "action": f"enter_{side}",
                "message": signal_message,
                "days_to_expiry": days_to_expiry,
                "estimated_atm_premium_per_share": premium,
                "estimated_atm_premium_per_contract": premium_per_contract,
            }
        )

    if not fired:
        pass

    result: Dict[str, Any] = {
        "messages": messages,
        "symbol": cfg.symbol,
        "configured_side": cfg.side,
        "selected_sides": selected_sides,
        "roc_lookback": cfg.roc_lookback,
        "vol_window": cfg.vol_window,
        "date": latest_date.date().isoformat(),
        "close": spot,
        "roc": latest_roc,
        # Backward-compatible field: pricing vol actually used for premium estimation (after floor).
        "pricing_vol_annualized": pricing_vol,
        # Explicit raw/latest calculated values from the latest bar.
        "latest_roc": latest_roc,
        "latest_pricing_vol_annualized": latest_pricing_vol,
        "fired_signals": fired_signals,
    }

    if "put" in selected_sides:
        result.update(
            {
                "put_roc_threshold": cfg.put_roc_threshold,
                "downside_vol_threshold": cfg.downside_vol_threshold,
                "downside_vol_annualized": latest_downside_vol,
                "put_trigger": put_trigger,
            }
        )
    if "call" in selected_sides:
        result.update(
            {
                "call_roc_threshold": cfg.call_roc_threshold,
                "upside_vol_threshold": cfg.upside_vol_threshold,
                "upside_vol_annualized": latest_upside_vol,
                "call_trigger": call_trigger,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily signal notifier: checks weekly reversal signal and prints put/call entry notification with estimated ATM premium."
    )
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol, e.g. SPY (used when --config is not set).")
    parser.add_argument("--side", choices=["put", "call", "both"], default="both", help="Which side(s) to evaluate.")
    parser.add_argument(
        "--config",
        default="option_signal_notifier.config",
        help=(
            "Optional config text file. One non-empty, non-comment line per setup using CLI-style flags, "
            "for example: --symbol SPY --side put --roc-lookback 18 --vol-window 20 --put-roc-threshold -0.005 --downside-vol-threshold 0.094507"
        ),
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "Optional local CSV path. If omitted, the script prefers data/etfs.csv cache and refreshes only missing rows "
            "from marketstack when cache is stale."
        ),
    )
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
        "--summary-json",
        default=None,
        help="Optional path to write machine-readable run summary JSON (keeps stdout logs unchanged).",
    )
    parser.add_argument(
        "--print-trades",
        type=int,
        default=0,
        help="Print recent rows with computed signals in single-config mode. Use -1 for all rows, 0 to disable.",
    )
    args = parser.parse_args()

    configs = _load_configs(args)

    # Load each symbol once so we refresh cache and fetch API data minimally.
    symbol_data: Dict[str, pd.DataFrame] = {}
    for symbol in sorted({cfg.symbol for cfg in configs}):
        symbol_data[symbol] = load_symbol_data(symbol, args.start_date, args.end_date, args.csv)

    all_messages: List[str] = []
    config_summaries: List[Dict[str, Any]] = []
    for idx, cfg in enumerate(configs, start=1):
        df = symbol_data[cfg.symbol]
        all_messages.append(f"\n=== Config #{idx} ===")
        evaluation = _evaluate_config(
            cfg=cfg,
            symbol_df=df,
            risk_free_rate=args.risk_free_rate,
            min_pricing_vol=args.min_pricing_vol,
            contract_size=args.contract_size,
        )
        all_messages.extend(evaluation["messages"])
        config_summaries.append({"config_index": idx, **evaluation})

    print("\n".join(all_messages).lstrip())

    if args.summary_json:
        ordered_config_summaries = sorted(
            config_summaries,
            key=lambda cfg: (
                not bool(cfg.get("call_trigger", False)),
                int(cfg.get("config_index", 0)),
            ),
        )
        total_signals = sum(len(cfg["fired_signals"]) for cfg in config_summaries)
        summary_payload = {
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "config_count": len(config_summaries),
            "signals_fired": total_signals > 0,
            "signal_count": total_signals,
            "configs": ordered_config_summaries,
        }
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_payload, indent=2))
        print(f"Wrote structured signal summary to {summary_path}.")

    if args.print_trades != 0 and len(configs) == 1:
        cfg = configs[0]
        signal_df = build_signal_frame(symbol_data[cfg.symbol], cfg.roc_lookback, cfg.vol_window)
        printable = signal_df.copy()
        printable["put_signal"] = (
            (printable["roc"] <= cfg.put_roc_threshold)
            & (printable["downside_vol_annualized"] >= cfg.downside_vol_threshold)
        )
        printable["call_signal"] = (
            (printable["roc"] >= cfg.call_roc_threshold)
            & (printable["upside_vol_annualized"] >= cfg.upside_vol_threshold)
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
    elif args.print_trades != 0:
        print("\n--print-trades is supported only with a single config row. Ignored for multi-config runs.")


if __name__ == "__main__":
    main()
