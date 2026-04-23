#!/usr/bin/env python3
"""
Model weekly ATM regime labels using ETF-level and same-price-neighborhood context.

Target labels come from data/_weekly_atm_worthless_only.csv:
- constraining_top
- constraining_low
- consolidating

Features are computed from data/etfs-all.csv on the entry date with
entry-open causality:
- underlying ETF metrics: price, price momentum, volume, volume momentum,
  price ROC, volume ROC, price stddev, volume stddev
- nearby ETF metrics: the same metrics averaged/std'd over neighboring ETFs
  by same-day price rank (k neighbors above + k neighbors below)
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_LABELS = ["constraining_top", "constraining_low", "consolidating"]


def default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count // 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict weekly ATM regimes from ETF + neighborhood features."
    )
    parser.add_argument(
        "--ohlcv-csv",
        default="data/etfs-all.csv",
        help="ETF OHLCV CSV (must include symbol,date,open,volume)",
    )
    parser.add_argument(
        "--weekly-csv",
        default="data/_weekly_atm_worthless_only.csv",
        help="Weekly labeled rows from weekly_atm_worthless_scan.py",
    )
    parser.add_argument(
        "--neighbors-k",
        type=int,
        default=5,
        help="How many price-neighbors above and below to use per date",
    )
    parser.add_argument(
        "--momentum-lookback",
        type=int,
        default=5,
        help="Lookback (days) for momentum features",
    )
    parser.add_argument(
        "--roc-lookback",
        type=int,
        default=1,
        help="Lookback (days) for ROC features",
    )
    parser.add_argument(
        "--std-lookback",
        type=int,
        default=10,
        help="Rolling window (days) for stddev features",
    )
    parser.add_argument(
        "--train-end-date",
        default=None,
        help=(
            "Optional YYYY-MM-DD cutoff for time split. "
            "Rows with entry_date <= cutoff are train, later rows are test."
        ),
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="If --train-end-date is omitted, use this trailing date fraction as test",
    )
    parser.add_argument(
        "--features-output-csv",
        default="data/weekly_regime_features.csv",
        help="Where to save merged training table",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="Number of worker processes for per-symbol feature engineering (default: half of available CPU cores)",
    )
    return parser.parse_args()


def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["symbol", "date", "open", "volume"])
    df["symbol"] = df["symbol"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["symbol", "date", "open", "volume"]).copy()
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_weekly(path: str) -> pd.DataFrame:
    cols = ["symbol", "entry_date", "expiry_date", "dte_calendar", "dte_trading", "regime", "outcome"]
    df = pd.read_csv(path, usecols=cols)
    df["symbol"] = df["symbol"].astype(str)
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df = df[df["regime"].isin(TARGET_LABELS)].dropna(subset=["symbol", "entry_date"]).copy()
    return df


def add_underlying_features(
    ohlcv: pd.DataFrame,
    momentum_lookback: int,
    roc_lookback: int,
    std_lookback: int,
    workers: int,
) -> pd.DataFrame:
    if workers < 1:
        raise ValueError("--workers must be >= 1")

    parts = [
        grp.copy()
        for _, grp in ohlcv.groupby("symbol", sort=False)
    ]

    args = [
        (part, momentum_lookback, roc_lookback, std_lookback)
        for part in parts
    ]

    if workers == 1:
        out_parts = [_add_underlying_features_for_symbol(a) for a in args]
    else:
        # Use process-based parallelism to spread symbol feature creation across cores.
        with get_context("spawn").Pool(processes=workers) as pool:
            out_parts = pool.map(_add_underlying_features_for_symbol, args)

    return pd.concat(out_parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)


def _add_underlying_features_for_symbol(
    args: tuple[pd.DataFrame, int, int, int]
) -> pd.DataFrame:
    df, momentum_lookback, roc_lookback, std_lookback = args
    df = df.sort_values("date").copy()

    # Entry-time safe: on entry_date we assume only today's open is known.
    # Price features are built from the open series.
    df["price"] = df["open"]
    df["price_momentum"] = df["open"].pct_change(momentum_lookback)
    df["price_roc"] = df["open"].pct_change(roc_lookback)
    df["price_stddev"] = df["open"].rolling(std_lookback, min_periods=std_lookback).std()

    # Volume for entry_date is not known at the open; use lagged volume only.
    vol_lag_1 = df["volume"].shift(1)
    vol_lag_m = df["volume"].shift(1 + momentum_lookback)
    vol_lag_r = df["volume"].shift(1 + roc_lookback)

    df["volume"] = vol_lag_1
    df["volume_momentum"] = (vol_lag_1 / vol_lag_m) - 1.0
    df["volume_roc"] = (vol_lag_1 / vol_lag_r) - 1.0
    df["volume_stddev"] = vol_lag_1.rolling(std_lookback, min_periods=std_lookback).std()

    return df


def add_neighbor_features(df: pd.DataFrame, neighbors_k: int, metric_cols: list[str]) -> pd.DataFrame:
    if neighbors_k < 1:
        raise ValueError("--neighbors-k must be >= 1")

    out = df.sort_values(["date", "price", "symbol"]).reset_index(drop=True).copy()
    by_date = out.groupby("date", sort=False)

    for col in metric_cols:
        sum_col = pd.Series(0.0, index=out.index)
        sumsq_col = pd.Series(0.0, index=out.index)
        cnt_col = pd.Series(0, index=out.index, dtype="int32")

        for step in range(1, neighbors_k + 1):
            for direction in (step, -step):
                shifted = by_date[col].shift(direction)
                valid = shifted.notna()
                vals = shifted.fillna(0.0)
                sum_col = sum_col + vals
                sumsq_col = sumsq_col + vals * vals
                cnt_col = cnt_col + valid.astype("int32")

        cnt_safe = cnt_col.replace(0, np.nan)
        mean_col = sum_col / cnt_safe
        var_col = (sumsq_col / cnt_safe) - (mean_col * mean_col)

        out[f"nbr_{col}_mean"] = mean_col
        out[f"nbr_{col}_std"] = np.sqrt(var_col.clip(lower=0))
        out[f"delta_{col}_vs_nbr_mean"] = out[col] - mean_col

    out["nbr_count"] = cnt_col
    return out


def build_feature_table(
    ohlcv: pd.DataFrame,
    weekly: pd.DataFrame,
    neighbors_k: int,
    momentum_lookback: int,
    roc_lookback: int,
    std_lookback: int,
    workers: int,
) -> pd.DataFrame:
    base = add_underlying_features(
        ohlcv=ohlcv,
        momentum_lookback=momentum_lookback,
        roc_lookback=roc_lookback,
        std_lookback=std_lookback,
        workers=workers,
    )

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

    with_neighbors = add_neighbor_features(base, neighbors_k=neighbors_k, metric_cols=metric_cols)

    keep_cols = ["symbol", "date"] + metric_cols + [
        f"nbr_{c}_mean" for c in metric_cols
    ] + [f"nbr_{c}_std" for c in metric_cols] + [
        f"delta_{c}_vs_nbr_mean" for c in metric_cols
    ] + ["nbr_count"]

    model_df = with_neighbors[keep_cols].copy()

    merged = weekly.merge(
        model_df,
        how="left",
        left_on=["symbol", "entry_date"],
        right_on=["symbol", "date"],
    ).drop(columns=["date"])

    return merged


def time_split(df: pd.DataFrame, train_end_date: str | None, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_end_date:
        cutoff = pd.to_datetime(train_end_date)
    else:
        unique_dates = np.array(sorted(df["entry_date"].dropna().unique()))
        if len(unique_dates) < 2:
            raise ValueError("Not enough unique dates to split train/test.")
        split_idx = int(np.floor((1.0 - test_fraction) * len(unique_dates)))
        split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
        cutoff = pd.Timestamp(unique_dates[split_idx - 1])

    train = df[df["entry_date"] <= cutoff].copy()
    test = df[df["entry_date"] > cutoff].copy()

    if train.empty or test.empty:
        raise ValueError(
            f"Bad split around cutoff {cutoff.date()}: train={len(train)}, test={len(test)}"
        )

    return train, test


def evaluate_baseline(y_train: pd.Series, y_test: pd.Series) -> dict[str, float]:
    labels = sorted(TARGET_LABELS)
    probs = y_train.value_counts(normalize=True).reindex(labels, fill_value=0.0)
    y_pred = pd.Series(np.repeat(probs.idxmax(), len(y_test)), index=y_test.index)
    y_prob = np.tile(probs.values, (len(y_test), 1))

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_prob, labels=labels),
    }


def evaluate_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[str, float], str, pd.Series]:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ],
        remainder="drop",
    )

    clf = Pipeline(
        steps=[
            ("pre", pre),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    n_jobs=None,
                ),
            ),
        ]
    )

    x_train = train[feature_cols]
    x_test = test[feature_cols]
    y_train = train["regime"]
    y_test = test["regime"]

    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prob = clf.predict_proba(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "log_loss": log_loss(y_test, prob, labels=clf.named_steps["model"].classes_),
    }

    report = classification_report(y_test, pred, digits=4)
    pred_series = pd.Series(pred, index=test.index, name="pred_regime")
    return metrics, report, pred_series


def main() -> None:
    args = parse_args()

    ohlcv = load_ohlcv(args.ohlcv_csv)
    weekly = load_weekly(args.weekly_csv)

    merged = build_feature_table(
        ohlcv=ohlcv,
        weekly=weekly,
        neighbors_k=args.neighbors_k,
        momentum_lookback=args.momentum_lookback,
        roc_lookback=args.roc_lookback,
        std_lookback=args.std_lookback,
        workers=args.workers,
    )
    merged = merged.replace([np.inf, -np.inf], np.nan)

    output_path = Path(args.features_output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    feature_cols = [
        c
        for c in merged.columns
        if c
        not in {
            "symbol",
            "entry_date",
            "expiry_date",
            "regime",
            "outcome",
        }
    ]

    train, test = time_split(merged, train_end_date=args.train_end_date, test_fraction=args.test_fraction)

    baseline_metrics = evaluate_baseline(train["regime"], test["regime"])
    model_metrics, report, pred = evaluate_model(train, test, feature_cols)

    per_etf = test[["symbol", "regime"]].copy()
    per_etf["pred_regime"] = pred.values
    per_etf["correct"] = (per_etf["regime"] == per_etf["pred_regime"]).astype(float)
    per_etf = (
        per_etf.groupby("symbol", as_index=False)
        .agg(
            test_rows=("correct", "size"),
            test_accuracy=("correct", "mean"),
        )
        .sort_values(["test_accuracy", "test_rows", "symbol"], ascending=[False, False, True])
    )

    print("Rows in merged table:", len(merged))
    print("Train rows:", len(train), "Test rows:", len(test))
    print("Train date range:", train["entry_date"].min().date(), "->", train["entry_date"].max().date())
    print("Test date range:", test["entry_date"].min().date(), "->", test["entry_date"].max().date())

    print("\nBaseline (predict by train appearance probabilities):")
    print(f"  accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"  log_loss: {baseline_metrics['log_loss']:.4f}")

    print("\nLogistic regression with ETF+neighbor features:")
    print(f"  accuracy: {model_metrics['accuracy']:.4f}")
    print(f"  log_loss: {model_metrics['log_loss']:.4f}")

    print("\nClassification report (model):")
    print(report)

    print("\nPer-ETF test accuracy (model):")
    print(per_etf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"Saved merged feature table: {output_path}")


if __name__ == "__main__":
    main()
