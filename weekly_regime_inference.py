#!/usr/bin/env python3
"""
Helpers for running latest per-symbol weekly regime inference from a saved model bundle.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from analyze_weekly_regime_with_etf_context import (
    add_neighbor_features,
    add_underlying_features,
    load_correlation_neighbors,
    load_ohlcv,
)


WEEKLY_REGIME_METRIC_COLS = [
    "price",
    "price_momentum",
    "volume",
    "volume_momentum",
    "price_roc",
    "volume_roc",
    "price_stddev",
    "volume_stddev",
    "high_open_over_open",
    "open_low_over_open",
    "high_close_over_close",
    "close_low_over_low",
    "high_low_over_high",
    "high_low_over_low",
]


def load_weekly_regime_bundle(model_path: str) -> Dict[str, Any]:
    bundle = joblib.load(model_path)
    if "feature_cols" not in bundle:
        raise ValueError(f"Weekly regime model at '{model_path}' is missing 'feature_cols'.")
    if "per_symbol_models" not in bundle:
        raise ValueError(f"Weekly regime model at '{model_path}' is missing 'per_symbol_models'.")
    return bundle


def predict_latest_weekly_regimes(
    symbols: Sequence[str],
    model_path: str,
    ohlcv_csv: str,
    correlation_pairs_csv: Optional[str] = None,
    as_of_date: Optional[str] = None,
    workers: int = 1,
) -> Dict[str, Dict[str, Any]]:
    target_symbols = sorted({str(s).upper() for s in symbols if str(s).strip()})
    if not target_symbols:
        return {}

    bundle = load_weekly_regime_bundle(model_path)
    feature_cols = list(bundle["feature_cols"])
    per_symbol_models = bundle["per_symbol_models"]

    corr_csv = correlation_pairs_csv or str(bundle.get("correlation_pairs_csv", ""))
    if not corr_csv:
        corr_csv = "data/etfs_correlation_pairs.03.csv"

    corr_threshold = float(bundle.get("correlation_threshold", 0.3))
    neighbor_map = load_correlation_neighbors(corr_csv, threshold=corr_threshold)

    required_symbols = set(target_symbols)
    for symbol in target_symbols:
        required_symbols.update(neighbor_map.get(symbol, set()))

    ohlcv = load_ohlcv(ohlcv_csv)
    ohlcv = ohlcv[ohlcv["symbol"].isin(required_symbols)].copy()
    if ohlcv.empty:
        raise ValueError(f"No OHLCV rows found in '{ohlcv_csv}' for requested symbols/neighbors.")

    if as_of_date:
        cutoff = pd.to_datetime(as_of_date)
        ohlcv = ohlcv[ohlcv["date"] <= cutoff].copy()
    if ohlcv.empty:
        raise ValueError("No OHLCV rows left after as-of-date filtering.")

    base = add_underlying_features(
        ohlcv=ohlcv,
        momentum_lookback=int(bundle.get("momentum_lookback", 5)),
        roc_lookback=int(bundle.get("roc_lookback", 1)),
        std_lookback=int(bundle.get("std_lookback", 10)),
        workers=max(1, int(workers)),
    )
    with_neighbors = add_neighbor_features(
        df=base,
        neighbor_map=neighbor_map,
        metric_cols=WEEKLY_REGIME_METRIC_COLS,
    ).replace([np.inf, -np.inf], np.nan)

    score_date = pd.Timestamp(with_neighbors["date"].max())
    day_rows = with_neighbors[with_neighbors["date"] == score_date].copy()
    day_rows["symbol"] = day_rows["symbol"].astype(str)
    day_rows = day_rows.set_index("symbol", drop=False)

    missing_features = [c for c in feature_cols if c not in day_rows.columns]
    if missing_features:
        raise ValueError(
            "Engineered features are missing columns required by the model: "
            + ", ".join(missing_features)
        )

    predictions: Dict[str, Dict[str, Any]] = {}
    for symbol in target_symbols:
        if symbol not in day_rows.index or symbol not in per_symbol_models:
            continue

        symbol_model_bundle = per_symbol_models[symbol]["model"]
        x = day_rows.loc[[symbol], feature_cols]

        if symbol_model_bundle.get("type") == "constant":
            pred_label = str(symbol_model_bundle["majority_label"])
            confidence = 1.0
        elif symbol_model_bundle.get("type") == "pipeline":
            pipeline = symbol_model_bundle["model"]
            pred_label = str(pipeline.predict(x)[0])
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(x)[0]
                confidence = float(np.max(probs))
            else:
                confidence = np.nan
        else:
            continue

        predictions[symbol] = {
            "regime_guess": pred_label,
            "regime_confidence": confidence,
            "regime_date": score_date.date().isoformat(),
        }

    return predictions
