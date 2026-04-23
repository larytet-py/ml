#!/usr/bin/env python3
"""
Backtest weekly short-option strategy driven by regime predictions.

Strategy for symbols SPY,GDX,IWM (customizable):
1) predicted constraining_top  -> sell call
2) predicted constraining_low  -> sell put
3) predicted consolidating     -> sell side chosen from historical frequency:
   if constraining_top count > constraining_low count, sell call; else sell put.

Expiry is always the last trading day of the same ISO week as entry.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd

from option_pricing import BlackScholesPricer

TRADING_DAYS_PER_YEAR = 252
PRICING_VOL_WINDOW_DAYS = 21
DEFAULT_SYMBOL_SOURCE_CSV = "data/etfs.csv"


@dataclass
class Trade:
    symbol: str
    predicted_regime: str
    side: str
    side_reason: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    scheduled_expiry_date: pd.Timestamp
    time_to_expiry_days: int
    entry_close: float
    exit_close: float
    strike: float
    premium: float
    intrinsic_at_expiry: float
    expired_itm: bool
    pnl_per_share: float
    pnl_per_contract: float
    pricing_vol: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest weekly short-call/put strategy from regime model predictions."
    )
    parser.add_argument("--ohlcv-csv", default="data/etfs-all.csv", help="OHLCV CSV path")
    parser.add_argument(
        "--features-csv",
        default="data/weekly_regime_features.csv",
        help="Feature table CSV with at least symbol,entry_date,expiry_date + model feature columns",
    )
    parser.add_argument(
        "--model",
        default="models/weekly_regime_model.joblib",
        help="Model artifact produced by analyze_weekly_regime_with_etf_context.py",
    )
    parser.add_argument(
        "--weekly-csv",
        default="data/weekly_atm_worthless_only.csv",
        help="Weekly sample CSV used to compute consolidation tie-break frequencies",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help=(
            "Comma-separated symbols to trade. "
            f"If omitted, use all symbols from {DEFAULT_SYMBOL_SOURCE_CSV}."
        ),
    )
    parser.add_argument("--start-date", default=None, help="Optional YYYY-MM-DD filter")
    parser.add_argument("--end-date", default=None, help="Optional YYYY-MM-DD filter")
    parser.add_argument(
        "--out-of-sample-only",
        action="store_true",
        help="Only include rows with entry_date > model train_end_date",
    )
    parser.add_argument("--risk-free-rate", type=float, default=0.04, help="Risk-free rate for Black-Scholes")
    parser.add_argument(
        "--min-pricing-vol",
        type=float,
        default=0.10,
        help="Vol floor (annualized) for premium pricing",
    )
    parser.add_argument("--contract-size", type=int, default=100, help="Shares per option contract")
    parser.add_argument("--allow-overlap", action="store_true", help="Allow overlapping weekly positions")
    parser.add_argument(
        "--consolidating-tie-side",
        choices=["put", "call"],
        default="put",
        help="If top/low frequencies are equal for a symbol, use this side",
    )
    parser.add_argument("--trades-out", default=None, help="Optional CSV output for trades")
    parser.add_argument(
        "--print-trades",
        type=int,
        default=20,
        help="How many recent trades to print. Use -1 for all, 0 to disable",
    )
    return parser.parse_args()


def _resolve_weekly_csv(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    fallback = Path("data/_weekly_atm_worthless_only.csv")
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError(
        f"Weekly sample CSV not found at '{path}' and fallback '{fallback}'."
    )


def infer_symbols_from_csv(path: str) -> list[str]:
    df = pd.read_csv(path, usecols=["symbol"])
    invalid_values = {"", "nan", "none", "null"}
    symbols = (
        df["symbol"]
        .astype(str)
        .str.strip()
        .loc[lambda s: ~s.str.lower().isin(invalid_values)]
        .dropna()
        .unique()
        .tolist()
    )
    symbols = sorted(symbols)
    if not symbols:
        raise ValueError(f"No symbols found in '{path}'.")
    return symbols


def load_ohlcv(ohlcv_csv: str, symbols: list[str], start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(ohlcv_csv, usecols=["symbol", "date", "open", "high", "low", "close", "volume"])
    df["symbol"] = df["symbol"].astype(str)
    df = df[df["symbol"].isin(symbols)].copy()
    if df.empty:
        raise ValueError("No OHLCV rows found for selected symbols.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "close"]).copy()

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No OHLCV rows left after date filters.")
    return df


def load_prediction_rows(
    features_csv: str,
    model_path: str,
    symbols: list[str],
    start_date: Optional[str],
    end_date: Optional[str],
    out_of_sample_only: bool,
) -> pd.DataFrame:
    model_bundle = joblib.load(model_path)
    feature_cols = list(model_bundle.get("feature_cols", []))
    per_symbol_models = model_bundle.get("per_symbol_models", {})
    if not feature_cols:
        raise ValueError("Model artifact missing 'feature_cols'.")

    usecols = ["symbol", "entry_date", "expiry_date"] + feature_cols
    df = pd.read_csv(features_csv, usecols=usecols)
    df["symbol"] = df["symbol"].astype(str)
    df = df[df["symbol"].isin(symbols)].copy()
    if df.empty:
        raise ValueError("No feature rows found for selected symbols.")

    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    df = df.dropna(subset=["entry_date", "expiry_date"]).copy()

    if start_date:
        df = df[df["entry_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["entry_date"] <= pd.to_datetime(end_date)]

    train_end_date = model_bundle.get("train_end_date")
    if out_of_sample_only and train_end_date:
        cutoff = pd.to_datetime(train_end_date)
        df = df[df["entry_date"] > cutoff]

    if df.empty:
        raise ValueError("No feature rows left after date/out-of-sample filters.")

    # Strategy trades weekly expiry only: keep the row with max expiry_date for each symbol+entry_date.
    df = (
        df.sort_values(["symbol", "entry_date", "expiry_date"])
        .groupby(["symbol", "entry_date"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    pred_parts: list[pd.DataFrame] = []
    skipped_missing_model: list[str] = []
    for symbol in symbols:
        s = df[df["symbol"] == symbol].copy()
        if s.empty:
            continue
        if symbol not in per_symbol_models:
            skipped_missing_model.append(symbol)
            continue

        symbol_model_bundle = per_symbol_models[symbol]["model"]
        if isinstance(symbol_model_bundle, dict):
            model_type = symbol_model_bundle.get("type")
            if model_type == "constant":
                majority = str(symbol_model_bundle["majority_label"])
                s["predicted_regime"] = [majority] * len(s)
            elif model_type == "pipeline":
                s["predicted_regime"] = symbol_model_bundle["model"].predict(s[feature_cols])
            else:
                raise ValueError(
                    f"Unsupported model bundle type '{model_type}' for symbol '{symbol}'."
                )
        else:
            s["predicted_regime"] = symbol_model_bundle.predict(s[feature_cols])
        pred_parts.append(s[["symbol", "entry_date", "predicted_regime"]])

    if not pred_parts:
        raise ValueError("No predictions were produced for selected symbols.")

    if skipped_missing_model:
        print(
            "Skipped symbols not present in model artifact: "
            f"{', '.join(sorted(skipped_missing_model)[:20])}"
            + (" ..." if len(skipped_missing_model) > 20 else "")
        )

    out = pd.concat(pred_parts, ignore_index=True)
    return out.sort_values(["symbol", "entry_date"]).reset_index(drop=True)


def consolidation_side_map(
    weekly_csv: str,
    symbols: list[str],
    tie_side: str,
) -> Dict[str, str]:
    df = pd.read_csv(_resolve_weekly_csv(weekly_csv), usecols=["symbol", "regime"])
    df["symbol"] = df["symbol"].astype(str)
    df = df[df["symbol"].isin(symbols)].copy()
    if df.empty:
        raise ValueError("No rows found in weekly sample CSV for selected symbols.")

    side_by_symbol: Dict[str, str] = {}
    for symbol in symbols:
        s = df[df["symbol"] == symbol]
        top_count = int((s["regime"] == "constraining_top").sum())
        low_count = int((s["regime"] == "constraining_low").sum())

        if top_count > low_count:
            side_by_symbol[symbol] = "call"
        elif low_count > top_count:
            side_by_symbol[symbol] = "put"
        else:
            side_by_symbol[symbol] = tie_side

    return side_by_symbol


def choose_side(predicted_regime: str, consolidating_side: str) -> tuple[str, str]:
    if predicted_regime == "constraining_top":
        return "call", "predicted_constraining_top"
    if predicted_regime == "constraining_low":
        return "put", "predicted_constraining_low"
    if predicted_regime == "consolidating":
        return consolidating_side, "predicted_consolidating_frequency_rule"
    raise ValueError(f"Unsupported predicted_regime '{predicted_regime}'.")


def build_trades(
    ohlcv: pd.DataFrame,
    predictions: pd.DataFrame,
    consolidating_side_by_symbol: Dict[str, str],
    risk_free_rate: float,
    min_pricing_vol_annualized: float,
    contract_size: int,
    allow_overlap: bool,
) -> pd.DataFrame:
    pricer = BlackScholesPricer(risk_free_rate=risk_free_rate, min_sigma=min_pricing_vol_annualized)

    all_trades: list[Trade] = []
    for symbol, sdf in ohlcv.groupby("symbol", sort=False):
        sdf = sdf.sort_values("date").reset_index(drop=True).copy()
        close = sdf["close"]
        returns = close.pct_change()
        sdf["pricing_vol_annualized"] = (
            returns.rolling(PRICING_VOL_WINDOW_DAYS).std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)
        )

        # Weekly expiry index for each row (same ISO week last trading day).
        iso = sdf["date"].dt.isocalendar()
        week_key = iso["year"].astype(int) * 100 + iso["week"].astype(int)
        week_last_idx = pd.Series(sdf.index, index=sdf.index).groupby(week_key).transform("max").astype(int)

        preds = predictions[predictions["symbol"] == symbol].copy()
        if preds.empty:
            continue
        pred_by_date = {
            pd.Timestamp(d).normalize(): r
            for d, r in zip(preds["entry_date"], preds["predicted_regime"])
        }

        next_entry_idx = 0
        for i, row in sdf.iterrows():
            if i < next_entry_idx:
                continue

            entry_date = pd.Timestamp(row["date"]).normalize()
            if entry_date not in pred_by_date:
                continue

            raw_sigma = row["pricing_vol_annualized"]
            if pd.isna(raw_sigma):
                continue

            predicted_regime = str(pred_by_date[entry_date])
            side, side_reason = choose_side(predicted_regime, consolidating_side_by_symbol[symbol])

            exit_idx = int(week_last_idx.iloc[i])
            if exit_idx >= len(sdf) or exit_idx <= i:
                # Skip same-day expiry in this EOD model.
                continue

            exit_row = sdf.iloc[exit_idx]
            exit_date = pd.Timestamp(exit_row["date"]).normalize()
            time_to_expiry_days = int((exit_date - entry_date).days)
            if time_to_expiry_days <= 0:
                continue

            entry_close = float(row["close"])
            exit_close = float(exit_row["close"])
            strike = entry_close
            t_years = time_to_expiry_days / 365.25

            pricing_vol = pricer.effective_sigma(float(raw_sigma))
            premium = pricer.price(
                side=side,
                spot=entry_close,
                strike=strike,
                time_to_expiry_years=t_years,
                sigma=float(raw_sigma),
            )
            intrinsic = pricer.intrinsic_value(side=side, strike=strike, spot=exit_close)
            pnl_per_share = premium - intrinsic

            all_trades.append(
                Trade(
                    symbol=symbol,
                    predicted_regime=predicted_regime,
                    side=side,
                    side_reason=side_reason,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    scheduled_expiry_date=exit_date,
                    time_to_expiry_days=time_to_expiry_days,
                    entry_close=entry_close,
                    exit_close=exit_close,
                    strike=strike,
                    premium=premium,
                    intrinsic_at_expiry=intrinsic,
                    expired_itm=intrinsic > 0.0,
                    pnl_per_share=pnl_per_share,
                    pnl_per_contract=pnl_per_share * contract_size,
                    pricing_vol=pricing_vol,
                )
            )

            if not allow_overlap:
                next_entry_idx = exit_idx + 1

    return pd.DataFrame([t.__dict__ for t in all_trades])


def summarize(trades_df: pd.DataFrame) -> None:
    if trades_df.empty:
        print("No trades generated with current settings.")
        return

    total = len(trades_df)
    wins = int((trades_df["pnl_per_share"] > 0).sum())
    itm = int(trades_df["expired_itm"].sum())
    total_pnl = float(trades_df["pnl_per_contract"].sum())
    avg_pnl = float(trades_df["pnl_per_contract"].mean())
    med_pnl = float(trades_df["pnl_per_contract"].median())

    print(f"Trades: {total}")
    print(f"Win rate: {wins / total:.2%}")
    print(f"Expired ITM: {itm} ({itm / total:.2%})")
    print(f"Total PnL (per 1 contract): {total_pnl:.2f}")
    print(f"Average PnL/trade (per 1 contract): {avg_pnl:.2f}")
    print(f"Median PnL/trade (per 1 contract): {med_pnl:.2f}")

    by_symbol_ordered = trades_df.sort_values(["symbol", "entry_date", "exit_date"]).copy()
    by_symbol_ordered["cum_pnl"] = by_symbol_ordered.groupby("symbol")["pnl_per_contract"].cumsum()
    by_symbol_ordered["cum_peak"] = by_symbol_ordered.groupby("symbol")["cum_pnl"].cummax()
    by_symbol_ordered["drawdown"] = by_symbol_ordered["cum_pnl"] - by_symbol_ordered["cum_peak"]
    by_symbol_drawdown = (
        by_symbol_ordered.groupby("symbol", as_index=False)
        .agg(max_drawdown=("drawdown", "min"))
    )

    print("\nBreakdown by symbol:")
    by_symbol = (
        trades_df.groupby("symbol", as_index=False)
        .agg(
            trades=("symbol", "size"),
            win_rate=("pnl_per_share", lambda s: float((s > 0).mean())),
            total_pnl=("pnl_per_contract", "sum"),
            avg_pnl=("pnl_per_contract", "mean"),
        )
        .merge(by_symbol_drawdown, on="symbol", how="left")
        .sort_values("symbol")
    )
    print(by_symbol.to_string(index=False, justify="center", float_format=lambda x: f"{x:.4f}"))

    print("\nBreakdown by predicted_regime -> sold side:")
    by_rule = (
        trades_df.groupby(["predicted_regime", "side"], as_index=False)
        .agg(
            trades=("side", "size"),
            win_rate=("pnl_per_share", lambda s: float((s > 0).mean())),
            total_pnl=("pnl_per_contract", "sum"),
        )
        .sort_values(["predicted_regime", "side"])
    )
    print(by_rule.to_string(index=False, justify="center", float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    args = parse_args()
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = infer_symbols_from_csv(DEFAULT_SYMBOL_SOURCE_CSV)
        print(f"--symbols not provided; using all symbols from {DEFAULT_SYMBOL_SOURCE_CSV}: {len(symbols)}")
    if not symbols:
        raise ValueError("No symbols resolved for backtest.")

    ohlcv = load_ohlcv(args.ohlcv_csv, symbols, args.start_date, args.end_date)
    predictions = load_prediction_rows(
        features_csv=args.features_csv,
        model_path=args.model,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        out_of_sample_only=args.out_of_sample_only,
    )
    symbols = sorted(predictions["symbol"].astype(str).unique().tolist())
    if not symbols:
        raise ValueError("No symbols with predictions after filtering/model compatibility checks.")
    consolidating_side_by_symbol = consolidation_side_map(
        weekly_csv=args.weekly_csv,
        symbols=symbols,
        tie_side=args.consolidating_tie_side,
    )

    print("Consolidating side by symbol (frequency rule):")
    for symbol in symbols:
        print(f"  {symbol}: {consolidating_side_by_symbol[symbol]}")

    trades_df = build_trades(
        ohlcv=ohlcv,
        predictions=predictions,
        consolidating_side_by_symbol=consolidating_side_by_symbol,
        risk_free_rate=args.risk_free_rate,
        min_pricing_vol_annualized=args.min_pricing_vol,
        contract_size=args.contract_size,
        allow_overlap=args.allow_overlap,
    )

    summarize(trades_df)

    if not trades_df.empty and args.print_trades != 0:
        cols = [
            "symbol",
            "predicted_regime",
            "side",
            "entry_date",
            "exit_date",
            "entry_close",
            "exit_close",
            "premium",
            "intrinsic_at_expiry",
            "expired_itm",
            "pnl_per_contract",
        ]
        to_show = trades_df[cols] if args.print_trades < 0 else trades_df[cols].tail(args.print_trades)
        print("\nRecent trades:")
        print(to_show.to_string(index=False, justify="center"))

    if args.trades_out:
        Path(args.trades_out).parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(args.trades_out, index=False)
        print(f"\nSaved trades to: {args.trades_out}")


if __name__ == "__main__":
    main()
