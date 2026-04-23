#!/usr/bin/env python3
"""
Model weekly ATM regime labels using ETF-level and correlation-neighborhood context.

Target labels come from data/_weekly_atm_worthless_only.csv:
- constraining_top
- constraining_low
- consolidating

Features are computed from data/etfs-all.csv on the entry date with
entry-open causality:
- underlying ETF metrics: price, price momentum, volume, volume momentum,
  price ROC, volume ROC, price stddev, volume stddev
- nearby ETF metrics: the same metrics averaged/std'd over neighboring ETFs
  selected by correlation pairs (corr > threshold or corr < -threshold)
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import get_context
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_LABELS = ["constraining_top", "constraining_low", "consolidating"]

# Read-only per-process caches used with Linux "fork" multiprocessing.
# Workers receive only symbol keys and look up immutable source frames here.
_UNDERLYING_SYMBOL_FRAMES: dict[str, pd.DataFrame] = {}
_EVAL_SYMBOL_FRAMES: dict[str, pd.DataFrame] = {}
_EVAL_FEATURE_COLS: tuple[str, ...] = ()


def log_phase(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {message}", flush=True)


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
        help="Deprecated: kept for backward compatibility, not used by correlation-neighbor logic",
    )
    parser.add_argument(
        "--correlation-pairs-csv",
        default="data/etfs-all_correlation_pairs.csv",
        help="Pairwise ETF correlation CSV (symbol_a,symbol_b,correlation)",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.3,
        help="Neighbor rule: use ETFs where correlation > threshold or < -threshold",
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
        "--coefficients-output-csv",
        default=None,
        help=(
            "Optional path to save per-symbol/class/feature coefficients as CSV "
            "(human-readable)"
        ),
    )
    parser.add_argument(
        "--coefficients-output-json",
        default="models/weekly_regime_model.json",
        help=(
            "Optional path to save per-symbol/class/feature coefficients as JSON "
            "(human-readable)"
        ),
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


def load_correlation_neighbors(path: str, threshold: float) -> dict[str, set[str]]:
    if threshold < 0:
        raise ValueError("--correlation-threshold must be >= 0")

    corr = pd.read_csv(path, usecols=["symbol_a", "symbol_b", "correlation"])
    corr["symbol_a"] = corr["symbol_a"].astype(str)
    corr["symbol_b"] = corr["symbol_b"].astype(str)
    corr["correlation"] = pd.to_numeric(corr["correlation"], errors="coerce")
    corr = corr.dropna(subset=["symbol_a", "symbol_b", "correlation"]).copy()

    filtered = corr[corr["correlation"].abs() > threshold].copy()
    if filtered.empty:
        return {}

    edges_ab = filtered[["symbol_a", "symbol_b"]].rename(
        columns={"symbol_a": "symbol", "symbol_b": "neighbor_symbol"}
    )
    edges_ba = filtered[["symbol_b", "symbol_a"]].rename(
        columns={"symbol_b": "symbol", "symbol_a": "neighbor_symbol"}
    )
    edges = pd.concat([edges_ab, edges_ba], ignore_index=True).drop_duplicates()
    edges = edges[edges["symbol"] != edges["neighbor_symbol"]]

    return (
        edges.groupby("symbol", sort=False)["neighbor_symbol"]
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )


def add_underlying_features(
    ohlcv: pd.DataFrame,
    momentum_lookback: int,
    roc_lookback: int,
    std_lookback: int,
    workers: int,
) -> pd.DataFrame:
    if workers < 1:
        raise ValueError("--workers must be >= 1")

    log_phase("Phase: underlying feature engineering started")
    symbol_frames = {
        symbol: grp.sort_values("date").reset_index(drop=True)
        for symbol, grp in ohlcv.groupby("symbol", sort=False)
    }
    args = [
        (symbol, momentum_lookback, roc_lookback, std_lookback)
        for symbol in symbol_frames.keys()
    ]
    global _UNDERLYING_SYMBOL_FRAMES
    _UNDERLYING_SYMBOL_FRAMES = symbol_frames

    if workers == 1:
        out_parts = [_add_underlying_features_for_symbol(a) for a in args]
    else:
        # On Linux, "fork" lets workers share parent memory copy-on-write.
        with get_context("fork").Pool(processes=workers) as pool:
            out_parts = pool.map(_add_underlying_features_for_symbol, args)

    _UNDERLYING_SYMBOL_FRAMES = {}
    result = pd.concat(out_parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    log_phase("Phase: underlying feature engineering completed")
    return result


def _add_underlying_features_for_symbol(
    args: tuple[str, int, int, int]
) -> pd.DataFrame:
    symbol, momentum_lookback, roc_lookback, std_lookback = args
    source_df = _UNDERLYING_SYMBOL_FRAMES[symbol]
    # Never mutate the shared source frame; build features on a worker-local copy.
    df = source_df.copy(deep=True)

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


def add_neighbor_features(
    df: pd.DataFrame,
    neighbor_map: dict[str, set[str]],
    metric_cols: list[str],
) -> pd.DataFrame:
    log_phase("Phase: neighbor feature engineering started")
    out = df.sort_values(["date", "symbol"]).reset_index(drop=True).copy()

    if not neighbor_map:
        for col in metric_cols:
            out[f"nbr_{col}_mean"] = np.nan
            out[f"nbr_{col}_std"] = np.nan
            out[f"delta_{col}_vs_nbr_mean"] = np.nan
        out["nbr_count"] = 0
        log_phase("Phase: neighbor feature engineering completed (no correlation neighbors found)")
        return out

    edge_rows: list[tuple[str, str]] = []
    for symbol, neighbors in neighbor_map.items():
        for neighbor_symbol in neighbors:
            edge_rows.append((symbol, neighbor_symbol))

    edges = pd.DataFrame(edge_rows, columns=["symbol", "neighbor_symbol"]).drop_duplicates()
    if edges.empty:
        for col in metric_cols:
            out[f"nbr_{col}_mean"] = np.nan
            out[f"nbr_{col}_std"] = np.nan
            out[f"delta_{col}_vs_nbr_mean"] = np.nan
        out["nbr_count"] = 0
        log_phase("Phase: neighbor feature engineering completed (no valid neighbor edges)")
        return out

    neighbors_daily = (
        out[["symbol", "date"]]
        .merge(edges, how="left", on="symbol")
    )

    neighbor_values = out[["symbol", "date"] + metric_cols].rename(
        columns={"symbol": "neighbor_symbol"}
    )
    neighbor_values["__neighbor_row_exists"] = 1
    neighbors_daily = neighbors_daily.merge(
        neighbor_values,
        how="left",
        on=["neighbor_symbol", "date"],
    )

    out_idx = out.set_index(["symbol", "date"])
    grp = neighbors_daily.groupby(["symbol", "date"], sort=False)
    out_idx["nbr_count"] = grp["__neighbor_row_exists"].sum(min_count=1).fillna(0).astype("int32")

    for col in metric_cols:
        means = grp[col].mean()
        stds = grp[col].std(ddof=0)
        out_idx[f"nbr_{col}_mean"] = means
        out_idx[f"nbr_{col}_std"] = stds
        out_idx[f"delta_{col}_vs_nbr_mean"] = out_idx[col] - means

    out = out_idx.reset_index()
    log_phase("Phase: neighbor feature engineering completed")
    return out


def build_feature_table(
    ohlcv: pd.DataFrame,
    weekly: pd.DataFrame,
    neighbor_map: dict[str, set[str]],
    momentum_lookback: int,
    roc_lookback: int,
    std_lookback: int,
    workers: int,
) -> pd.DataFrame:
    log_phase("Phase: build feature table started")
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

    with_neighbors = add_neighbor_features(
        base,
        neighbor_map=neighbor_map,
        metric_cols=metric_cols,
    )

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

    log_phase("Phase: build feature table completed")
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


def _fit_and_evaluate_single_symbol(
    args: tuple[str, str | None, float]
) -> dict:
    symbol, train_end_date, test_fraction = args
    source_df = _EVAL_SYMBOL_FRAMES[symbol]
    # Never mutate the shared source frame; split/train from a worker-local copy.
    symbol_df = source_df.copy(deep=True)
    feature_cols = list(_EVAL_FEATURE_COLS)
    log_phase(f"Per-symbol training started: {symbol}")
    train, test = time_split(symbol_df, train_end_date=train_end_date, test_fraction=test_fraction)

    train_probs = train["regime"].value_counts(normalize=True)
    model_bundle = fit_multinomial_head(train, feature_cols)
    pred = predict_multinomial_head(model_bundle, test[feature_cols])
    pred_series = pd.Series(pred, index=test.index)
    true_series = test["regime"].astype(str)

    per_symbol_label_rows: list[dict] = []
    per_symbol_precision_recall_rows: list[dict] = []
    per_symbol_confusion_rows: list[dict] = []

    for label in TARGET_LABELS:
        mask = test["regime"] == label
        support = int(mask.sum())
        if support > 0:
            model_class_accuracy = float((pred_series[mask] == label).mean())
        else:
            model_class_accuracy = np.nan

        baseline_class_accuracy = float(train_probs.get(label, 0.0))
        per_symbol_label_rows.append(
            {
                "symbol": symbol,
                "label": label,
                "support": support,
                "probability_guess_class_accuracy": baseline_class_accuracy,
                "model_class_accuracy": model_class_accuracy,
                "accuracy_lift": (
                    model_class_accuracy - baseline_class_accuracy
                    if not np.isnan(model_class_accuracy)
                    else np.nan
                ),
            }
        )

        true_positive = int(((true_series == label) & (pred_series == label)).sum())
        predicted_count = int((pred_series == label).sum())
        precision = (
            float(true_positive / predicted_count)
            if predicted_count > 0
            else np.nan
        )
        recall = (
            float(true_positive / support)
            if support > 0
            else np.nan
        )

        per_symbol_precision_recall_rows.append(
            {
                "symbol": symbol,
                "label": label,
                "support_true": support,
                "predicted_count": predicted_count,
                "true_positive": true_positive,
                "precision": precision,
                "recall": recall,
            }
        )

    for true_label in TARGET_LABELS:
        true_mask = true_series == true_label
        for predicted_label in TARGET_LABELS:
            count = int((true_mask & (pred_series == predicted_label)).sum())
            per_symbol_confusion_rows.append(
                {
                    "symbol": symbol,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": count,
                }
            )

    result = {
        "symbol": symbol,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "per_symbol_label_rows": per_symbol_label_rows,
        "per_symbol_precision_recall_rows": per_symbol_precision_recall_rows,
        "per_symbol_confusion_rows": per_symbol_confusion_rows,
        "model_entry": {
            "model": model_bundle,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_date_min": pd.Timestamp(train["entry_date"].min()).date().isoformat(),
            "train_date_max": pd.Timestamp(train["entry_date"].max()).date().isoformat(),
            "test_date_min": pd.Timestamp(test["entry_date"].min()).date().isoformat(),
            "test_date_max": pd.Timestamp(test["entry_date"].max()).date().isoformat(),
        },
    }
    log_phase(f"Per-symbol training completed: {symbol}")
    return result


def evaluate_per_symbol_multinomial_models(
    merged: pd.DataFrame,
    feature_cols: list[str],
    train_end_date: str | None,
    test_fraction: float,
    workers: int,
) -> tuple[pd.DataFrame, dict]:
    if workers < 1:
        raise ValueError("--workers must be >= 1")

    log_phase("Phase: per-symbol model training/evaluation started")
    per_symbol_label_rows: list[dict] = []
    per_symbol_precision_recall_rows: list[dict] = []
    per_symbol_confusion_rows: list[dict] = []
    per_symbol_models: dict[str, dict] = {}
    train_rows_total = 0
    test_rows_total = 0

    symbol_frames = {
        symbol: symbol_df.sort_values("entry_date").reset_index(drop=True)
        for symbol, symbol_df in merged.groupby("symbol", sort=True)
    }
    symbol_jobs = [
        (symbol, train_end_date, test_fraction)
        for symbol in symbol_frames.keys()
    ]
    global _EVAL_SYMBOL_FRAMES, _EVAL_FEATURE_COLS
    _EVAL_SYMBOL_FRAMES = symbol_frames
    _EVAL_FEATURE_COLS = tuple(feature_cols)

    if workers == 1:
        symbol_results = [_fit_and_evaluate_single_symbol(job) for job in symbol_jobs]
    else:
        # On Linux, "fork" lets workers share parent memory copy-on-write.
        with get_context("fork").Pool(processes=workers) as pool:
            symbol_results = pool.map(_fit_and_evaluate_single_symbol, symbol_jobs)

    _EVAL_SYMBOL_FRAMES = {}
    _EVAL_FEATURE_COLS = ()
    for result in symbol_results:
        train_rows_total += result["train_rows"]
        test_rows_total += result["test_rows"]
        per_symbol_label_rows.extend(result["per_symbol_label_rows"])
        per_symbol_precision_recall_rows.extend(result["per_symbol_precision_recall_rows"])
        per_symbol_confusion_rows.extend(result["per_symbol_confusion_rows"])
        per_symbol_models[result["symbol"]] = result["model_entry"]

    if not per_symbol_label_rows:
        raise ValueError("No symbols available for training/evaluation.")

    per_symbol_label_table = (
        pd.DataFrame(per_symbol_label_rows)
        .sort_values(["symbol", "label"], ascending=[True, True])
    )
    per_symbol_precision_recall_table = (
        pd.DataFrame(per_symbol_precision_recall_rows)
        .sort_values(["symbol", "label"], ascending=[True, True])
    )
    per_symbol_confusion_table = (
        pd.DataFrame(per_symbol_confusion_rows)
        .sort_values(["symbol", "true_label", "predicted_label"], ascending=[True, True, True])
    )

    split_stats = {
        "train_rows": int(train_rows_total),
        "test_rows": int(test_rows_total),
        "train_date_min": min(v["train_date_min"] for v in per_symbol_models.values()),
        "train_date_max": max(v["train_date_max"] for v in per_symbol_models.values()),
        "test_date_min": min(v["test_date_min"] for v in per_symbol_models.values()),
        "test_date_max": max(v["test_date_max"] for v in per_symbol_models.values()),
    }
    log_phase("Phase: per-symbol model training/evaluation completed")
    return per_symbol_label_table, {
        "per_symbol_models": per_symbol_models,
        "split_stats": split_stats,
        "per_symbol_precision_recall": per_symbol_precision_recall_table,
        "per_symbol_confusion": per_symbol_confusion_table,
    }


def extract_coefficients_long(
    per_symbol_models: dict[str, dict],
    feature_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for symbol, symbol_model in sorted(per_symbol_models.items()):
        model_bundle = symbol_model["model"]
        if model_bundle["type"] != "pipeline":
            rows.append(
                {
                    "symbol": symbol,
                    "class_label": model_bundle["majority_label"],
                    "feature": "__constant_prediction__",
                    "coefficient": np.nan,
                    "intercept": np.nan,
                    "model_type": model_bundle["type"],
                }
            )
            continue

        clf = model_bundle["model"]
        lr = clf.named_steps["model"]
        classes = [str(c) for c in lr.classes_]

        for class_idx, class_label in enumerate(classes):
            intercept = float(lr.intercept_[class_idx])
            for feat_idx, feat_name in enumerate(feature_cols):
                rows.append(
                    {
                        "symbol": symbol,
                        "class_label": class_label,
                        "feature": feat_name,
                        "coefficient": float(lr.coef_[class_idx, feat_idx]),
                        "intercept": intercept,
                        "model_type": model_bundle["type"],
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    log_phase("Pipeline started")
    args = parse_args()

    log_phase(f"Phase: load OHLCV started ({args.ohlcv_csv})")
    ohlcv = load_ohlcv(args.ohlcv_csv)
    log_phase(f"Phase: load OHLCV completed (rows={len(ohlcv)})")
    log_phase(f"Phase: load weekly labels started ({args.weekly_csv})")
    weekly = load_weekly(args.weekly_csv)
    log_phase(f"Phase: load weekly labels completed (rows={len(weekly)})")
    log_phase(
        f"Phase: load correlation pairs started ({args.correlation_pairs_csv}, threshold={args.correlation_threshold})"
    )
    neighbor_map = load_correlation_neighbors(
        path=args.correlation_pairs_csv,
        threshold=args.correlation_threshold,
    )
    log_phase(
        f"Phase: load correlation pairs completed (symbols_with_neighbors={len(neighbor_map)})"
    )

    merged = build_feature_table(
        ohlcv=ohlcv,
        weekly=weekly,
        neighbor_map=neighbor_map,
        momentum_lookback=args.momentum_lookback,
        roc_lookback=args.roc_lookback,
        std_lookback=args.std_lookback,
        workers=args.workers,
    )
    merged = merged.replace([np.inf, -np.inf], np.nan)
    log_phase(f"Phase: clean infinities completed (rows={len(merged)})")

    output_path = Path(args.features_output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_phase(f"Phase: save merged features started ({output_path})")
    merged.to_csv(output_path, index=False)
    log_phase(f"Phase: save merged features completed ({output_path})")

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

    per_symbol_label_table, eval_bundle = evaluate_per_symbol_multinomial_models(
        merged=merged,
        feature_cols=feature_cols,
        train_end_date=args.train_end_date,
        test_fraction=args.test_fraction,
        workers=args.workers,
    )
    split_stats = eval_bundle["split_stats"]
    precision_recall_table = eval_bundle["per_symbol_precision_recall"]
    confusion_table = eval_bundle["per_symbol_confusion"]

    print("Per-symbol per-label class accuracy (historical-probability guess vs model):")
    print(per_symbol_label_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("Per-symbol per-label precision/recall:")
    print(precision_recall_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("Per-symbol confusion matrices (rows=true label, cols=predicted label):")
    for symbol in sorted(confusion_table["symbol"].unique()):
        symbol_conf = confusion_table[confusion_table["symbol"] == symbol]
        matrix = symbol_conf.pivot(
            index="true_label",
            columns="predicted_label",
            values="count",
        ).reindex(index=TARGET_LABELS, columns=TARGET_LABELS, fill_value=0)
        print()
        print(symbol)
        print(matrix.to_string())

    model_output_path = Path(args.model_output)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "modeling_mode": "per_symbol_multinomial",
        "per_symbol_models": eval_bundle["per_symbol_models"],
        "feature_cols": feature_cols,
        "target_labels": TARGET_LABELS,
        "neighbors_k": args.neighbors_k,
        "correlation_pairs_csv": args.correlation_pairs_csv,
        "correlation_threshold": args.correlation_threshold,
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
    log_phase(f"Phase: save model artifact started ({model_output_path})")
    joblib.dump(bundle, model_output_path)
    log_phase(f"Phase: save model artifact completed ({model_output_path})")

    metadata_output_path = Path(args.metadata_output_json)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_output": str(model_output_path),
        "target_labels": TARGET_LABELS,
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "neighbors_k": args.neighbors_k,
        "correlation_pairs_csv": args.correlation_pairs_csv,
        "correlation_threshold": args.correlation_threshold,
        "momentum_lookback": args.momentum_lookback,
        "roc_lookback": args.roc_lookback,
        "std_lookback": args.std_lookback,
        "modeling_mode": "per_symbol_multinomial",
        "train_rows": int(split_stats["train_rows"]),
        "test_rows": int(split_stats["test_rows"]),
        "train_date_min": split_stats["train_date_min"],
        "train_date_max": split_stats["train_date_max"],
        "test_date_min": split_stats["test_date_min"],
        "test_date_max": split_stats["test_date_max"],
        "per_symbol_per_label_class_accuracy": per_symbol_label_table.to_dict(orient="records"),
        "per_symbol_per_label_precision_recall": precision_recall_table.to_dict(orient="records"),
        "per_symbol_confusion_matrix_long": confusion_table.to_dict(orient="records"),
    }
    log_phase(f"Phase: save metadata started ({metadata_output_path})")
    metadata_output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log_phase(f"Phase: save metadata completed ({metadata_output_path})")

    coefficients_df = extract_coefficients_long(
        per_symbol_models=eval_bundle["per_symbol_models"],
        feature_cols=feature_cols,
    )
    if args.coefficients_output_csv:
        coefficients_csv_path = Path(args.coefficients_output_csv)
        coefficients_csv_path.parent.mkdir(parents=True, exist_ok=True)
        log_phase(f"Phase: save coefficients CSV started ({coefficients_csv_path})")
        coefficients_df.to_csv(coefficients_csv_path, index=False)
        log_phase(f"Phase: save coefficients CSV completed ({coefficients_csv_path})")
        print(f"Saved coefficients CSV: {coefficients_csv_path}")
    if args.coefficients_output_json:
        coefficients_json_path = Path(args.coefficients_output_json)
        coefficients_json_path.parent.mkdir(parents=True, exist_ok=True)
        log_phase(f"Phase: save coefficients JSON started ({coefficients_json_path})")
        coefficients_json_path.write_text(
            json.dumps(coefficients_df.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )
        log_phase(f"Phase: save coefficients JSON completed ({coefficients_json_path})")
        print(f"Saved coefficients JSON: {coefficients_json_path}")

    print(f"Saved merged feature table: {output_path}")
    print(f"Saved model artifact: {model_output_path}")
    print(f"Saved model metadata: {metadata_output_path}")
    log_phase("Pipeline completed")


if __name__ == "__main__":
    main()
