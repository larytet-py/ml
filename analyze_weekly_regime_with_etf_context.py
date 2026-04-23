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
import json
import os
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
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
        "--model-output",
        default="models/weekly_regime_model.joblib",
        help="Where to save the trained sklearn pipeline + feature schema",
    )
    parser.add_argument(
        "--metadata-output-json",
        default="models/weekly_regime_model_metadata.json",
        help="Where to save training metadata JSON (human-readable)",
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
    majority_label = y_train.value_counts().idxmax()
    y_pred = pd.Series(np.repeat(majority_label, len(y_test)), index=y_test.index)
    return {"accuracy": accuracy_score(y_test, y_pred)}


def build_numeric_pipeline(feature_cols: list[str]) -> Pipeline:
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

    return Pipeline(
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

def fit_multinomial_head(
    train: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    y_train = train["regime"]
    majority_label = y_train.value_counts().idxmax()

    # If a symbol has only one class in train, fallback to constant label prediction.
    if y_train.nunique() < 2:
        return {
            "type": "constant",
            "majority_label": majority_label,
        }

    clf = build_numeric_pipeline(feature_cols)
    clf.fit(train[feature_cols], y_train)
    return {
        "type": "pipeline",
        "majority_label": majority_label,
        "model": clf,
    }


def predict_multinomial_head(model_bundle: dict, x: pd.DataFrame) -> np.ndarray:
    if model_bundle["type"] == "constant":
        return np.repeat(model_bundle["majority_label"], len(x))
    return model_bundle["model"].predict(x)


def evaluate_per_symbol_multinomial_models(
    merged: pd.DataFrame,
    feature_cols: list[str],
    train_end_date: str | None,
    test_fraction: float,
) -> tuple[dict[str, float], dict[str, float], str, pd.DataFrame, pd.DataFrame, dict]:
    prediction_parts: list[pd.DataFrame] = []
    per_symbol_rows: list[dict] = []
    per_symbol_models: dict[str, dict] = {}
    train_rows_total = 0
    test_rows_total = 0

    for symbol, symbol_df in merged.groupby("symbol", sort=True):
        train, test = time_split(symbol_df, train_end_date=train_end_date, test_fraction=test_fraction)
        train_rows_total += len(train)
        test_rows_total += len(test)

        baseline_pred = train["regime"].value_counts().idxmax()
        model_bundle = fit_multinomial_head(train, feature_cols)
        pred = predict_multinomial_head(model_bundle, test[feature_cols])

        per_symbol_rows.append(
            {
                "symbol": symbol,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "baseline_test_accuracy": float((test["regime"] == baseline_pred).mean()),
                "model_test_accuracy": float((test["regime"].to_numpy() == pred).mean()),
            }
        )

        prediction_parts.append(
            pd.DataFrame(
                {
                    "symbol": test["symbol"].to_numpy(),
                    "entry_date": test["entry_date"].to_numpy(),
                    "regime": test["regime"].to_numpy(),
                    "pred_regime": pred,
                    "baseline_pred_regime": np.repeat(baseline_pred, len(test)),
                }
            )
        )

        per_symbol_models[symbol] = {
            "model": model_bundle,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_date_min": pd.Timestamp(train["entry_date"].min()).date().isoformat(),
            "train_date_max": pd.Timestamp(train["entry_date"].max()).date().isoformat(),
            "test_date_min": pd.Timestamp(test["entry_date"].min()).date().isoformat(),
            "test_date_max": pd.Timestamp(test["entry_date"].max()).date().isoformat(),
        }

    if not prediction_parts:
        raise ValueError("No symbols available for training/evaluation.")

    pred_df = pd.concat(prediction_parts, ignore_index=True)
    pred_df["model_correct"] = (pred_df["regime"] == pred_df["pred_regime"]).astype(float)
    pred_df["baseline_correct"] = (pred_df["regime"] == pred_df["baseline_pred_regime"]).astype(float)

    model_metrics = {
        "accuracy": float(accuracy_score(pred_df["regime"], pred_df["pred_regime"])),
    }
    baseline_metrics = {
        "accuracy": float(accuracy_score(pred_df["regime"], pred_df["baseline_pred_regime"])),
    }

    report = classification_report(pred_df["regime"], pred_df["pred_regime"], digits=4)
    per_symbol_table = (
        pd.DataFrame(per_symbol_rows)
        .assign(accuracy_lift=lambda d: d["model_test_accuracy"] - d["baseline_test_accuracy"])
        .sort_values(["model_test_accuracy", "test_rows", "symbol"], ascending=[False, False, True])
    )
    per_symbol_per_label_rows: list[dict] = []
    for symbol, grp in pred_df.groupby("symbol", sort=True):
        precision, recall, f1, support = precision_recall_fscore_support(
            grp["regime"],
            grp["pred_regime"],
            labels=TARGET_LABELS,
            zero_division=0,
        )
        for i, label in enumerate(TARGET_LABELS):
            per_symbol_per_label_rows.append(
                {
                    "symbol": symbol,
                    "label": label,
                    "support": int(support[i]),
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1_score": float(f1[i]),
                }
            )
    per_symbol_per_label_table = pd.DataFrame(per_symbol_per_label_rows).sort_values(
        ["symbol", "label"], ascending=[True, True]
    )

    split_stats = {
        "train_rows": int(train_rows_total),
        "test_rows": int(test_rows_total),
        "train_date_min": min(v["train_date_min"] for v in per_symbol_models.values()),
        "train_date_max": max(v["train_date_max"] for v in per_symbol_models.values()),
        "test_date_min": min(v["test_date_min"] for v in per_symbol_models.values()),
        "test_date_max": max(v["test_date_max"] for v in per_symbol_models.values()),
    }
    return baseline_metrics, model_metrics, report, pred_df, per_symbol_table, per_symbol_per_label_table, {
        "per_symbol_models": per_symbol_models,
        "split_stats": split_stats,
    }


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

    baseline_metrics, model_metrics, report, pred_df, per_symbol_table, per_symbol_per_label_table, eval_bundle = (
        evaluate_per_symbol_two_head_models(
            merged=merged,
            feature_cols=feature_cols,
            train_end_date=args.train_end_date,
            test_fraction=args.test_fraction,
        )
    )
    split_stats = eval_bundle["split_stats"]

    print("Rows in merged table:", len(merged))
    print("Train rows:", split_stats["train_rows"], "Test rows:", split_stats["test_rows"])
    print("Train date range:", split_stats["train_date_min"], "->", split_stats["train_date_max"])
    print("Test date range:", split_stats["test_date_min"], "->", split_stats["test_date_max"])
    print("Modeling mode: per-symbol with separate binary heads for consolidating and constraining_top")

    print("\nBaseline (per-symbol train appearance probabilities):")
    print(f"  accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"  log_loss: {baseline_metrics['log_loss']:.4f}")

    print("\nPer-symbol two-head logistic model:")
    print(f"  accuracy: {model_metrics['accuracy']:.4f}")
    print(f"  log_loss: {model_metrics['log_loss']:.4f}")

    print("\nClassification report (model):")
    print(report)

    print("\nPer-symbol test accuracy (baseline vs model):")
    print(per_symbol_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPer-symbol per-label metrics (model):")
    print(per_symbol_per_label_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    model_output_path = Path(args.model_output)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "modeling_mode": "per_symbol_two_head",
        "binary_head_labels": BINARY_HEAD_LABELS,
        "per_symbol_models": eval_bundle["per_symbol_models"],
        "feature_cols": feature_cols,
        "target_labels": TARGET_LABELS,
        "neighbors_k": args.neighbors_k,
        "momentum_lookback": args.momentum_lookback,
        "roc_lookback": args.roc_lookback,
        "std_lookback": args.std_lookback,
        "train_rows": int(split_stats["train_rows"]),
        "test_rows": int(split_stats["test_rows"]),
        "train_end_date": (
            args.train_end_date
            if args.train_end_date
            else split_stats["train_date_max"]
        ),
    }
    joblib.dump(bundle, model_output_path)

    metadata_output_path = Path(args.metadata_output_json)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_output": str(model_output_path),
        "target_labels": TARGET_LABELS,
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "neighbors_k": args.neighbors_k,
        "momentum_lookback": args.momentum_lookback,
        "roc_lookback": args.roc_lookback,
        "std_lookback": args.std_lookback,
        "modeling_mode": "per_symbol_two_head",
        "binary_head_labels": BINARY_HEAD_LABELS,
        "train_rows": int(split_stats["train_rows"]),
        "test_rows": int(split_stats["test_rows"]),
        "train_date_min": split_stats["train_date_min"],
        "train_date_max": split_stats["train_date_max"],
        "test_date_min": split_stats["test_date_min"],
        "test_date_max": split_stats["test_date_max"],
        "baseline_accuracy": float(baseline_metrics["accuracy"]),
        "baseline_log_loss": float(baseline_metrics["log_loss"]),
        "model_accuracy": float(model_metrics["accuracy"]),
        "model_log_loss": float(model_metrics["log_loss"]),
        "per_symbol_metrics": per_symbol_table.to_dict(orient="records"),
        "per_symbol_per_label_metrics": per_symbol_per_label_table.to_dict(orient="records"),
    }
    metadata_output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved merged feature table: {output_path}")
    print(f"Saved model artifact: {model_output_path}")
    print(f"Saved model metadata: {metadata_output_path}")


if __name__ == "__main__":
    main()
