#!/usr/bin/env python3
"""
Daily ETF scanner that loads a trained weekly regime model and emits alerts.

The model artifact is produced by analyze_weekly_regime_with_etf_context.py.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from analyze_weekly_regime_with_etf_context import add_neighbor_features, add_underlying_features, load_ohlcv


TARGET_LABELS = ["constraining_top", "constraining_low", "consolidating"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score ETFs daily for high-probability weekly regime opportunities."
    )
    parser.add_argument(
        "--ohlcv-csv",
        default="data/etfs-all.csv",
        help="ETF OHLCV CSV (must include symbol,date,open,volume)",
    )
    parser.add_argument(
        "--model-path",
        default="models/weekly_regime_model.joblib",
        help="Saved model bundle from analyze_weekly_regime_with_etf_context.py",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Optional YYYY-MM-DD date to score. Defaults to latest date in CSV.",
    )
    parser.add_argument(
        "--alert-regime",
        choices=TARGET_LABELS,
        default="consolidating",
        help="Which regime probability to use for alerting.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Alert when probability(alert_regime) >= threshold.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for feature generation.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/daily_worthless_alerts.csv",
        help="Where to save full daily scores.",
    )
    parser.add_argument(
        "--alerts-csv",
        default="data/daily_worthless_alerts_only.csv",
        help="Where to save threshold-passing alerts only.",
    )
    return parser.parse_args()


def load_model_bundle(path: str) -> dict:
    bundle = joblib.load(path)
    required = ["model", "feature_cols"]
    missing = [k for k in required if k not in bundle]
    if missing:
        raise ValueError(f"Model bundle missing required keys: {missing}")
    return bundle


def build_daily_rows(
    ohlcv: pd.DataFrame,
    as_of_date: pd.Timestamp,
    neighbors_k: int,
    momentum_lookback: int,
    roc_lookback: int,
    std_lookback: int,
    workers: int,
) -> pd.DataFrame:
    metric_cols = [
        "price",
        "price_momentum",
        "volume",
        "volume_momentum",
        "price_roc",
        "volume_roc",
        "price_stddev",
        "volume_stddev",
    ]

    base = add_underlying_features(
        ohlcv=ohlcv,
        momentum_lookback=momentum_lookback,
        roc_lookback=roc_lookback,
        std_lookback=std_lookback,
        workers=workers,
    )
    with_neighbors = add_neighbor_features(base, neighbors_k=neighbors_k, metric_cols=metric_cols)
    day_rows = with_neighbors[with_neighbors["date"] == as_of_date].copy()
    if day_rows.empty:
        raise ValueError(f"No rows found for as-of date {as_of_date.date().isoformat()}.")
    return day_rows.sort_values("symbol").reset_index(drop=True)


def classify_trade_idea(regime: str) -> str:
    if regime == "consolidating":
        return "both-side short premium candidate"
    if regime == "constraining_top":
        return "call-side short premium candidate"
    if regime == "constraining_low":
        return "put-side short premium candidate"
    return "unknown"


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(args.model_path)

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    neighbors_k = int(bundle.get("neighbors_k", 5))
    momentum_lookback = int(bundle.get("momentum_lookback", 5))
    roc_lookback = int(bundle.get("roc_lookback", 1))
    std_lookback = int(bundle.get("std_lookback", 10))

    ohlcv = load_ohlcv(args.ohlcv_csv)

    if args.as_of_date:
        as_of_date = pd.to_datetime(args.as_of_date)
        ohlcv = ohlcv[ohlcv["date"] <= as_of_date].copy()
    else:
        as_of_date = pd.Timestamp(ohlcv["date"].max())

    if ohlcv.empty:
        raise ValueError("No OHLCV rows available after date filtering.")

    day_rows = build_daily_rows(
        ohlcv=ohlcv,
        as_of_date=as_of_date,
        neighbors_k=neighbors_k,
        momentum_lookback=momentum_lookback,
        roc_lookback=roc_lookback,
        std_lookback=std_lookback,
        workers=args.workers,
    )
    day_rows = day_rows.replace([np.inf, -np.inf], np.nan)

    missing_features = [c for c in feature_cols if c not in day_rows.columns]
    if missing_features:
        raise ValueError(
            "Daily features are missing columns required by the model: "
            + ", ".join(missing_features)
        )

    x = day_rows[feature_cols]
    pred = model.predict(x)
    prob = model.predict_proba(x)
    classes = [str(c) for c in model.named_steps["model"].classes_]

    scored = day_rows[["symbol", "date", "open", "price"]].copy()
    scored["pred_regime"] = pred
    scored["pred_confidence"] = np.max(prob, axis=1)

    for idx, cls in enumerate(classes):
        scored[f"prob_{cls}"] = prob[:, idx]

    if args.alert_regime not in classes:
        raise ValueError(
            f"Alert regime '{args.alert_regime}' not in model classes: {classes}"
        )

    alert_col = f"prob_{args.alert_regime}"
    scored["alert_regime"] = args.alert_regime
    scored["alert_probability"] = scored[alert_col]
    scored["trade_idea"] = scored["pred_regime"].map(classify_trade_idea)
    scored["call_strike_up"] = scored["open"].map(lambda x: int(math.ceil(float(x))))
    scored["put_strike_down"] = scored["open"].map(lambda x: int(math.floor(float(x))))
    scored["is_alert"] = scored["alert_probability"] >= args.threshold

    scored = scored.sort_values(["is_alert", "alert_probability", "symbol"], ascending=[False, False, True])

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_csv, index=False)

    alerts = scored[scored["is_alert"]].copy()
    alerts_csv = Path(args.alerts_csv)
    alerts_csv.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(alerts_csv, index=False)

    print(f"As-of date: {as_of_date.date().isoformat()}")
    print(f"Scored ETFs: {len(scored)}")
    print(f"Alert regime: {args.alert_regime}")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Alerts: {len(alerts)}")

    show_cols = [
        "symbol",
        "pred_regime",
        "pred_confidence",
        "alert_probability",
        "trade_idea",
        "call_strike_up",
        "put_strike_down",
    ]
    if alerts.empty:
        print("\nNo alerts above threshold.")
    else:
        print("\nTop alerts:")
        print(alerts[show_cols].head(25).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nSaved full scores: {output_csv}")
    print(f"Saved alerts only: {alerts_csv}")


if __name__ == "__main__":
    main()
