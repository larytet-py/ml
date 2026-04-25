#!/usr/bin/env python3
import math
from typing import Dict, TypedDict

import numpy as np
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


class EntryCandidate(TypedDict):
    exit_idx: int
    days_to_friday: int
    trend_vol_signal: float


def compute_weekly_entry_candidates(
    signal_df: pd.DataFrame,
    side: str,
    put_roc_threshold: float,
    call_roc_threshold: float,
    downside_vol_threshold_annualized: float,
    upside_vol_threshold_annualized: float,
    allow_overlap: bool,
    require_future_week_row: bool = True,
) -> Dict[int, EntryCandidate]:
    if side not in {"put", "call"}:
        raise ValueError(f"Unsupported side '{side}'. Expected 'put' or 'call'.")

    n = len(signal_df)
    if n == 0:
        return {}

    entries: Dict[int, EntryCandidate] = {}
    next_entry_idx = 0
    blocked_until: np.datetime64 | None = None

    date_series = signal_df["date"]
    date_values = date_series.to_numpy()
    weekday = date_series.dt.weekday.to_numpy(dtype=int, copy=False)
    days_to_friday = 4 - weekday

    roc = signal_df["roc"].to_numpy(dtype=float, copy=False)
    pricing_vol = signal_df["pricing_vol_annualized"].to_numpy(dtype=float, copy=False)

    # Use the last trading row within the same ISO week as expiry.
    iso = date_series.dt.isocalendar()
    week_key = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    week_last_idx = (
        pd.Series(signal_df.index, index=signal_df.index)
        .groupby(week_key)
        .transform("max")
        .to_numpy(dtype=int, copy=False)
    )

    base_mask = (~np.isnan(roc)) & (~np.isnan(pricing_vol)) & (days_to_friday > 0)
    if side == "put":
        trend = signal_df["downside_vol_annualized"].to_numpy(dtype=float, copy=False)
        trigger_mask = base_mask & (~np.isnan(trend)) & (roc <= put_roc_threshold) & (trend >= downside_vol_threshold_annualized)
    else:
        trend = signal_df["upside_vol_annualized"].to_numpy(dtype=float, copy=False)
        trigger_mask = base_mask & (~np.isnan(trend)) & (roc >= call_roc_threshold) & (trend >= upside_vol_threshold_annualized)

    for i in np.flatnonzero(trigger_mask):
        if not allow_overlap:
            if require_future_week_row:
                if i < next_entry_idx:
                    continue
            else:
                row_date = date_values[i]
                if blocked_until is not None and row_date <= blocked_until:
                    continue

        exit_idx = int(week_last_idx[i])
        if require_future_week_row:
            if exit_idx >= n or exit_idx <= i:
                continue

        d2f = int(days_to_friday[i])
        entries[int(i)] = {
            "exit_idx": exit_idx,
            "days_to_friday": d2f,
            "trend_vol_signal": float(trend[i]),
        }

        if not allow_overlap:
            if require_future_week_row:
                next_entry_idx = exit_idx + 1
            else:
                blocked_until = date_values[i] + np.timedelta64(d2f, "D")

    return entries
