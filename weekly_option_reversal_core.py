#!/usr/bin/env python3
import math
from typing import Dict, Literal

import pandas as pd


TRADING_DAYS_PER_YEAR = 252
PRICING_VOL_WINDOW_DAYS = 21


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


def build_signal_frame(df: pd.DataFrame, roc_lookback: int, vol_window: int) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    returns = close.pct_change()
    out["roc"] = close / close.shift(roc_lookback) - 1.0
    out["downside_vol_annualized"] = returns.rolling(vol_window).apply(downside_std, raw=True) * math.sqrt(TRADING_DAYS_PER_YEAR)
    out["upside_vol_annualized"] = returns.rolling(vol_window).apply(upside_std, raw=True) * math.sqrt(TRADING_DAYS_PER_YEAR)
    # Keep pricing sigma on a fixed 21-trading-day realized-vol estimate.
    out["pricing_vol_annualized"] = returns.rolling(PRICING_VOL_WINDOW_DAYS).std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)
    return out


def compute_weekly_entry_candidates(
    signal_df: pd.DataFrame,
    side: Literal["put", "call"],
    put_roc_threshold: float,
    call_roc_threshold: float,
    downside_vol_threshold_annualized: float,
    upside_vol_threshold_annualized: float,
    allow_overlap: bool,
) -> Dict[int, Dict[str, float]]:
    if side not in {"put", "call"}:
        raise ValueError(f"Unsupported side '{side}'. Expected 'put' or 'call'.")

    entries: Dict[int, Dict[str, float]] = {}
    next_entry_idx = 0

    # Use the last trading row within the same ISO week as expiry.
    iso = signal_df["date"].dt.isocalendar()
    week_key = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    week_last_idx = pd.Series(signal_df.index, index=signal_df.index).groupby(week_key).transform("max").astype(int)

    for i in range(len(signal_df)):
        if i < next_entry_idx:
            continue

        row = signal_df.iloc[i]
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

        entry_date = pd.Timestamp(row["date"])
        days_to_friday = 4 - entry_date.weekday()
        if days_to_friday <= 0:
            # Skip same-day expiry signals (e.g., Friday close) in this EOD model.
            continue

        exit_idx = int(week_last_idx.iloc[i])
        if exit_idx >= len(signal_df):
            continue
        if exit_idx <= i:
            continue

        entries[i] = {
            "exit_idx": float(exit_idx),
            "days_to_friday": float(days_to_friday),
            "trend_vol_signal": trend_vol_signal,
        }

        if not allow_overlap:
            next_entry_idx = exit_idx + 1

    return entries
