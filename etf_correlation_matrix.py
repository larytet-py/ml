#!/usr/bin/env python3
"""Build a full cross-symbol correlation matrix from ETF OHLCV data."""

from __future__ import annotations

import argparse
import math
import os
import time
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd


def log_phase(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {message}", flush=True)


def default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count // 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create pairwise symbol correlations from data/etfs-all.csv "
            "using multiprocessing."
        )
    )
    parser.add_argument(
        "--input-csv",
        default="data/etfs-all.csv",
        help="Input CSV path with at least symbol,date and a numeric value column",
    )
    parser.add_argument(
        "--value-col",
        default="close",
        help="Numeric column used to compute correlation (default: close)",
    )
    parser.add_argument(
        "--output-matrix-csv",
        default="data/etfs-all_correlation_matrix.csv",
        help="Output CSV path for square correlation matrix",
    )
    parser.add_argument(
        "--output-pairs-csv",
        default="data/etfs-all_correlation_pairs.csv",
        help="Output CSV path for long pairwise correlations",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=2,
        help="Minimum overlapping dates required to compute a pairwise correlation",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="Number of worker processes (default: half of available CPU cores)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=25,
        help="How many left-side symbols each worker task processes",
    )
    parser.add_argument(
        "--last-days",
        type=int,
        default=30,
        help="Only use rows from the last N days (based on latest date in input)",
    )
    return parser.parse_args()


def load_and_pivot(input_csv: str, value_col: str) -> pd.DataFrame:
    use_cols = ["symbol", "date", value_col]
    df = pd.read_csv(input_csv, usecols=use_cols)
    df["symbol"] = df["symbol"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["symbol", "date", value_col]).copy()

    pivot = df.pivot_table(
        index="date",
        columns="symbol",
        values=value_col,
        aggfunc="last",
    )
    pivot = pivot.sort_index().sort_index(axis=1)
    return pivot


def build_jobs(symbol_count: int, chunk_size: int) -> list[tuple[int, int]]:
    jobs: list[tuple[int, int]] = []
    upper = max(0, symbol_count - 1)
    for start in range(0, upper, chunk_size):
        jobs.append((start, min(start + chunk_size, upper)))
    return jobs


def pair_count_for_job(symbol_count: int, start: int, end: int) -> int:
    # Sum of (symbol_count - i - 1) for i in [start, end).
    first = symbol_count - start - 1
    last = symbol_count - end
    length = end - start
    return (length * (first + last)) // 2


def corr_from_overlap(x: np.ndarray, y: np.ndarray, min_overlap: int) -> tuple[float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    overlap = int(mask.sum())
    if overlap < min_overlap:
        return math.nan, overlap

    xv = x[mask]
    yv = y[mask]
    x_std = float(np.std(xv))
    y_std = float(np.std(yv))
    if x_std == 0.0 or y_std == 0.0:
        return math.nan, overlap

    corr = float(np.corrcoef(xv, yv)[0, 1])
    return corr, overlap


def worker_pairwise(
    args: tuple[np.ndarray, list[str], int, int, int, int]
) -> tuple[list[tuple[str, str, float, int]], int]:
    values, symbols, start_i, end_i, min_overlap, pair_count = args
    n_symbols = len(symbols)
    out: list[tuple[str, str, float, int]] = []

    for i in range(start_i, end_i):
        xi = values[:, i]
        si = symbols[i]
        for j in range(i + 1, n_symbols):
            corr, overlap = corr_from_overlap(xi, values[:, j], min_overlap)
            out.append((si, symbols[j], corr, overlap))

    return out, pair_count


def compute_pairwise(
    pivot: pd.DataFrame,
    min_overlap: int,
    workers: int,
    chunk_size: int,
) -> pd.DataFrame:
    symbols = [str(s) for s in pivot.columns.tolist()]
    values = pivot.to_numpy(dtype=np.float64)
    jobs = build_jobs(len(symbols), chunk_size)

    if not jobs:
        return pd.DataFrame(columns=["symbol_a", "symbol_b", "correlation", "overlap_days"])

    packed_jobs = [
        (values, symbols, start, end, min_overlap, pair_count_for_job(len(symbols), start, end))
        for start, end in jobs
    ]
    total_pairs = len(symbols) * (len(symbols) - 1) // 2
    done_pairs = 0
    last_log = time.monotonic()

    def maybe_log_progress(force: bool = False) -> None:
        nonlocal last_log
        now = time.monotonic()
        if force or (now - last_log >= 1.0):
            pct = (100.0 * done_pairs / total_pairs) if total_pairs else 100.0
            log_phase(f"Progress: {pct:6.2f}% ({done_pairs:,}/{total_pairs:,} pairs)")
            last_log = now

    if workers == 1:
        chunked_results = []
        for job in packed_jobs:
            chunk, pair_count = worker_pairwise(job)
            chunked_results.append(chunk)
            done_pairs += pair_count
            maybe_log_progress()
    else:
        with get_context("spawn").Pool(processes=workers) as pool:
            chunked_results = []
            for chunk, pair_count in pool.imap_unordered(worker_pairwise, packed_jobs):
                chunked_results.append(chunk)
                done_pairs += pair_count
                maybe_log_progress()

    maybe_log_progress(force=True)

    rows = [row for chunk in chunked_results for row in chunk]
    return pd.DataFrame(rows, columns=["symbol_a", "symbol_b", "correlation", "overlap_days"])


def to_matrix(symbols: list[str], pairs: pd.DataFrame) -> pd.DataFrame:
    n = len(symbols)
    matrix = np.full((n, n), np.nan, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)

    index_by_symbol = {symbol: i for i, symbol in enumerate(symbols)}

    for row in pairs.itertuples(index=False):
        i = index_by_symbol[row.symbol_a]
        j = index_by_symbol[row.symbol_b]
        matrix[i, j] = row.correlation
        matrix[j, i] = row.correlation

    return pd.DataFrame(matrix, index=symbols, columns=symbols)


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if args.min_overlap < 2:
        raise ValueError("--min-overlap must be >= 2")
    if args.last_days < 1:
        raise ValueError("--last-days must be >= 1")

    log_phase(f"Loading input CSV: {args.input_csv}")
    pivot = load_and_pivot(args.input_csv, args.value_col)
    max_date = pivot.index.max()
    cutoff = max_date - pd.Timedelta(days=args.last_days)
    pivot = pivot[pivot.index >= cutoff].copy()

    symbols = [str(s) for s in pivot.columns.tolist()]
    log_phase(
        f"Prepared pivot table with {len(pivot):,} dates and {len(symbols):,} symbols"
    )
    log_phase(
        f"Date window: last {args.last_days} day(s), cutoff={cutoff.date()}, max_date={max_date.date()}"
    )

    log_phase(
        f"Computing pairwise correlations with workers={args.workers}, chunk_size={args.chunk_size}"
    )
    pairs = compute_pairwise(
        pivot=pivot,
        min_overlap=args.min_overlap,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )

    log_phase(f"Building square matrix from {len(pairs):,} symbol pairs")
    matrix = to_matrix(symbols, pairs)

    ensure_parent(args.output_pairs_csv)
    ensure_parent(args.output_matrix_csv)

    pairs.to_csv(args.output_pairs_csv, index=False)
    matrix.to_csv(args.output_matrix_csv, index=True)

    log_phase(f"Saved pairwise correlations to: {args.output_pairs_csv}")
    log_phase(f"Saved correlation matrix to: {args.output_matrix_csv}")


if __name__ == "__main__":
    main()
