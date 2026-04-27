#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

TRADING_DAYS_PER_YEAR = 252
EPS = 1e-12
SYMBOL_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]
WINDOW_KEYS = ["roc_window", "accel_roc_window", "accel_shift_window", "vol_window", "corr_window"]

DEFAULT_CONFIG = {
    "window_ranges": {
        "roc_window": {"min": 2, "max": 60, "step": 1},
        "accel_roc_window": {"min": 1, "max": 20, "step": 1},
        "accel_shift_window": {"min": 1, "max": 10, "step": 1},
        "vol_window": {"min": 2, "max": 40, "step": 1},
        "corr_window": {"min": 5, "max": 90, "step": 1},
    },
    "smoothing": {
        "accel_ema_window": 8,
        "vol_ema_window": 8,
    },
    "price_momentum_source": "close",
}

_GLOBAL_SYMBOL_FRAMES: Optional[Dict[str, pd.DataFrame]] = None
_GLOBAL_ACCEL_ROC_VALUES: Optional[List[int]] = None
_GLOBAL_ACCEL_SHIFT_VALUES: Optional[List[int]] = None
_GLOBAL_RET_CTX: Optional[Tuple[Dict[pd.Timestamp, float], Dict[pd.Timestamp, int]]] = None
_GLOBAL_ACCEL_CTX_MAP: Optional[
    Dict[Tuple[int, int], Tuple[Dict[pd.Timestamp, float], Dict[pd.Timestamp, int]]]
] = None
_GLOBAL_CFG: Optional[Dict[str, object]] = None
_GLOBAL_WINDOW_VALUES: Optional[Dict[str, List[int]]] = None


def _log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[features][{ts}] {message}", flush=True)


def _downside_std(arr: np.ndarray) -> float:
    v = arr[arr < 0]
    if len(v) < 2:
        return 0.0
    return float(v.std(ddof=0))


def _upside_std(arr: np.ndarray) -> float:
    v = arr[arr > 0]
    if len(v) < 2:
        return 0.0
    return float(v.std(ddof=0))


def _load_generator_config(path: Optional[str]) -> Dict[str, object]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if not path:
        return cfg

    loaded = yaml.safe_load(Path(path).read_text()) or {}
    if isinstance(loaded.get("window_ranges"), dict):
        cfg["window_ranges"].update(loaded["window_ranges"])
    if isinstance(loaded.get("smoothing"), dict):
        cfg["smoothing"].update(loaded["smoothing"])
    if "price_momentum_source" in loaded:
        cfg["price_momentum_source"] = str(loaded["price_momentum_source"]).strip().lower()
    return cfg


def _parse_window_values(cfg: Dict[str, object]) -> Dict[str, List[int]]:
    ranges = cfg["window_ranges"]
    out: Dict[str, List[int]] = {}
    for key in WINDOW_KEYS:
        lo = int(ranges[key]["min"])
        hi = int(ranges[key]["max"])
        step = int(ranges[key].get("step", 1))
        out[key] = list(range(lo, hi + 1, step))
    return out


def _worker_count(workers: int) -> int:
    if workers == 0:
        return os.cpu_count() or 1
    if workers < 0:
        return max(1, (os.cpu_count() or 1) + workers + 1)
    return max(1, workers)


def _prepare_input_df(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    rename_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df[SYMBOL_COLUMNS].copy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).drop_duplicates(subset=["symbol", "date"], keep="last")
    return df


def _empty_accel_ctx(
    accel_roc_values: List[int], accel_shift_values: List[int]
) -> Dict[Tuple[int, int], Tuple[Dict[pd.Timestamp, float], Dict[pd.Timestamp, int]]]:
    return {(ar, sh): ({}, {}) for ar in accel_roc_values for sh in accel_shift_values}


def _merge_sum_count_maps(
    dst_sum: Dict[pd.Timestamp, float],
    dst_count: Dict[pd.Timestamp, int],
    src_sum: Dict[pd.Timestamp, float],
    src_count: Dict[pd.Timestamp, int],
) -> None:
    for dt, val in src_sum.items():
        dst_sum[dt] = dst_sum.get(dt, 0.0) + float(val)
    for dt, cnt in src_count.items():
        dst_count[dt] = dst_count.get(dt, 0) + int(cnt)


def _symbol_context_partials(
    sdf: pd.DataFrame,
    accel_roc_values: List[int],
    accel_shift_values: List[int],
) -> Tuple[
    Dict[pd.Timestamp, float],
    Dict[pd.Timestamp, int],
    Dict[Tuple[int, int], Tuple[Dict[pd.Timestamp, float], Dict[pd.Timestamp, int]]],
]:
    ret_sum: Dict[pd.Timestamp, float] = {}
    ret_count: Dict[pd.Timestamp, int] = {}
    accel_ctx = _empty_accel_ctx(accel_roc_values, accel_shift_values)

    dates = sdf["date"]
    ret_close = sdf["close"].pct_change(1)
    for dt, rv in zip(dates, ret_close):
        if pd.isna(rv):
            continue
        ret_sum[dt] = ret_sum.get(dt, 0.0) + float(rv)
        ret_count[dt] = ret_count.get(dt, 0) + 1

    for ar in accel_roc_values:
        base = sdf["close"].pct_change(ar)
        for sh in accel_shift_values:
            accel = base - base.shift(sh)
            sum_map, count_map = accel_ctx[(ar, sh)]
            for dt, av in zip(dates, accel):
                if pd.isna(av):
                    continue
                sum_map[dt] = sum_map.get(dt, 0.0) + float(av)
                count_map[dt] = count_map.get(dt, 0) + 1

    return ret_sum, ret_count, accel_ctx


def _init_context_worker(
    symbol_frames: Dict[str, pd.DataFrame],
    accel_roc_values: List[int],
    accel_shift_values: List[int],
) -> None:
    global _GLOBAL_SYMBOL_FRAMES, _GLOBAL_ACCEL_ROC_VALUES, _GLOBAL_ACCEL_SHIFT_VALUES
    _GLOBAL_SYMBOL_FRAMES = symbol_frames
    _GLOBAL_ACCEL_ROC_VALUES = accel_roc_values
    _GLOBAL_ACCEL_SHIFT_VALUES = accel_shift_values


def _context_worker(symbol: str) -> Tuple[
    Dict[pd.Timestamp, float],
    Dict[pd.Timestamp, int],
    Dict[Tuple[int, int], Tuple[Dict[pd.Timestamp, float], Dict[pd.Timestamp, int]]],
]:
    assert _GLOBAL_SYMBOL_FRAMES is not None
    assert _GLOBAL_ACCEL_ROC_VALUES is not None
    assert _GLOBAL_ACCEL_SHIFT_VALUES is not None
    sdf = _GLOBAL_SYMBOL_FRAMES[symbol]
    return _symbol_context_partials(sdf, _GLOBAL_ACCEL_ROC_VALUES, _GLOBAL_ACCEL_SHIFT_VALUES)


def _compute_symbol_wide(
    sdf: pd.DataFrame,
    ret_ctx: Tuple[Dict[pd.Timestamp, float], Dict[pd.Timestamp, int]],
    accel_ctx_map: Dict[Tuple[int, int], Tuple[Dict[pd.Timestamp, float], Dict[pd.Timestamp, int]]],
    window_values: Dict[str, List[int]],
    accel_ema_window: int,
    vol_ema_window: int,
    price_momentum_source: str,
) -> pd.DataFrame:
    out = sdf.copy()
    ret_1d_close = out["close"].pct_change(1)
    ret_1d_open = out["open"].pct_change(1)
    ret_close = ret_1d_close
    ret_open = ret_1d_open

    ret_sum_map, ret_count_map = ret_ctx
    ret_sum = out["date"].map(ret_sum_map)
    ret_count = out["date"].map(ret_count_map)
    denom_ret = (ret_count - 1).where((ret_count - 1) > 0, np.nan)
    market_ex_self_ret = (ret_sum - ret_1d_close) / denom_ret

    feature_map: Dict[str, pd.Series] = {
        "ret_1d_close": ret_1d_close,
        "ret_1d_open": ret_1d_open,
        "day_range_over_open": (out["high"] - out["low"]) / out["open"],
        "close_position_in_range": (out["close"] - out["low"]) / ((out["high"] - out["low"]) + EPS),
        "body_over_open": (out["close"] - out["open"]) / out["open"],
    }

    for w in window_values["roc_window"]:
        volume_momentum = (out["volume"].shift(1) / out["volume"].shift(1 + w)) - 1.0
        feature_map[f"roc_close_w{w}"] = out["close"].pct_change(w)
        feature_map[f"roc_open_w{w}"] = out["open"].pct_change(w)
        feature_map[f"price_momentum_w{w}"] = out[price_momentum_source].pct_change(w)
        feature_map[f"volume_momentum_w{w}"] = volume_momentum
        feature_map[f"volume_roc_w{w}"] = volume_momentum

    for ar in window_values["accel_roc_window"]:
        close_roc = out["close"].pct_change(ar)
        open_roc = out["open"].pct_change(ar)
        for sh in window_values["accel_shift_window"]:
            accel_close = close_roc - close_roc.shift(sh)
            accel_open = open_roc - open_roc.shift(sh)
            feature_map[f"accel_close_w{ar}_s{sh}"] = accel_close
            feature_map[f"accel_open_w{ar}_s{sh}"] = accel_open
            feature_map[f"accel_close_ema_w{ar}_s{sh}"] = accel_close.ewm(span=accel_ema_window, adjust=False).mean()
            feature_map[f"accel_open_ema_w{ar}_s{sh}"] = accel_open.ewm(span=accel_ema_window, adjust=False).mean()
            feature_map[f"accel_regime_sign_w{ar}_s{sh}"] = np.sign(accel_close).rolling(ar).mean().abs()

            acc_sum_map, acc_count_map = accel_ctx_map[(ar, sh)]
            acc_sum = out["date"].map(acc_sum_map)
            acc_count = out["date"].map(acc_count_map)
            acc_denom = (acc_count - 1).where((acc_count - 1) > 0, np.nan)
            market_ex_self_accel = (acc_sum - accel_close) / acc_denom
            for cw in window_values["corr_window"]:
                feature_map[f"accel_corr_market_ar{ar}_sh{sh}_cw{cw}"] = accel_close.rolling(cw).corr(
                    market_ex_self_accel
                )

    for vw in window_values["vol_window"]:
        rv_close = ret_close.rolling(vw).std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)
        rv_open = ret_open.rolling(vw).std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)
        dvol = ret_close.rolling(vw).apply(_downside_std, raw=True) * math.sqrt(TRADING_DAYS_PER_YEAR)
        uvol = ret_close.rolling(vw).apply(_upside_std, raw=True) * math.sqrt(TRADING_DAYS_PER_YEAR)
        feature_map[f"realized_vol_close_w{vw}"] = rv_close
        feature_map[f"realized_vol_open_w{vw}"] = rv_open
        feature_map[f"downside_vol_w{vw}"] = dvol
        feature_map[f"upside_vol_w{vw}"] = uvol
        feature_map[f"realized_vol_close_ema_w{vw}"] = rv_close.ewm(span=vol_ema_window, adjust=False).mean()
        feature_map[f"realized_vol_open_ema_w{vw}"] = rv_open.ewm(span=vol_ema_window, adjust=False).mean()
        feature_map[f"downside_vol_ema_w{vw}"] = dvol.ewm(span=vol_ema_window, adjust=False).mean()
        feature_map[f"upside_vol_ema_w{vw}"] = uvol.ewm(span=vol_ema_window, adjust=False).mean()
        feature_map[f"price_stddev_w{vw}"] = out["close"].rolling(vw).std(ddof=0)
        feature_map[f"volume_stddev_w{vw}"] = out["volume"].shift(1).rolling(vw).std(ddof=0)

    for cw in window_values["corr_window"]:
        feature_map[f"corr_market_cw{cw}"] = ret_close.rolling(cw).corr(market_ex_self_ret)

    features_df = pd.DataFrame(feature_map, index=out.index)
    return pd.concat([out[SYMBOL_COLUMNS].reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)


def _init_wide_worker(
    symbol_frames: Dict[str, pd.DataFrame],
    ret_ctx: Tuple[Dict[pd.Timestamp, float], Dict[pd.Timestamp, int]],
    accel_ctx_map: Dict[Tuple[int, int], Tuple[Dict[pd.Timestamp, float], Dict[pd.Timestamp, int]]],
    cfg: Dict[str, object],
    window_values: Dict[str, List[int]],
) -> None:
    global _GLOBAL_SYMBOL_FRAMES, _GLOBAL_RET_CTX, _GLOBAL_ACCEL_CTX_MAP, _GLOBAL_CFG, _GLOBAL_WINDOW_VALUES
    _GLOBAL_SYMBOL_FRAMES = symbol_frames
    _GLOBAL_RET_CTX = ret_ctx
    _GLOBAL_ACCEL_CTX_MAP = accel_ctx_map
    _GLOBAL_CFG = cfg
    _GLOBAL_WINDOW_VALUES = window_values


def _wide_worker(symbol: str) -> Tuple[str, pd.DataFrame, str, str]:
    assert _GLOBAL_SYMBOL_FRAMES is not None
    assert _GLOBAL_RET_CTX is not None
    assert _GLOBAL_ACCEL_CTX_MAP is not None
    assert _GLOBAL_CFG is not None
    assert _GLOBAL_WINDOW_VALUES is not None

    sdf = _GLOBAL_SYMBOL_FRAMES[symbol]
    smoothing = _GLOBAL_CFG["smoothing"]
    wide = _compute_symbol_wide(
        sdf=sdf,
        ret_ctx=_GLOBAL_RET_CTX,
        accel_ctx_map=_GLOBAL_ACCEL_CTX_MAP,
        window_values=_GLOBAL_WINDOW_VALUES,
        accel_ema_window=int(smoothing["accel_ema_window"]),
        vol_ema_window=int(smoothing["vol_ema_window"]),
        price_momentum_source=str(_GLOBAL_CFG["price_momentum_source"]),
    )
    date_min = str(wide["date"].min().date())
    date_max = str(wide["date"].max().date())
    return symbol, wide, date_min, date_max


def _write_schema(path: Path, sample_df: pd.DataFrame) -> None:
    schema = {"columns": [{"name": c, "dtype": str(sample_df[c].dtype)} for c in sample_df.columns]}
    path.write_text(json.dumps(schema, indent=2, sort_keys=True))


def _write_meta(
    path: Path,
    total_rows: int,
    symbol_count: int,
    date_min: Optional[str],
    date_max: Optional[str],
    universe: List[str],
    cfg: Dict[str, object],
    workers: int,
) -> None:
    meta = {
        "row_count": int(total_rows),
        "symbol_count": int(symbol_count),
        "universe_size": int(len(universe)),
        "date_min": date_min,
        "date_max": date_max,
        "feature_version": "3.1-wide-parquet",
        "workers": int(workers),
        "window_ranges": cfg["window_ranges"],
        "smoothing": cfg["smoothing"],
        "price_momentum_source": cfg["price_momentum_source"],
        "output_format": "parquet",
    }
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def build_features(
    input_csv: str,
    output_parquet: str,
    schema_json: str,
    meta_json: str,
    window_config_yaml: Optional[str],
    force_rebuild: bool,
    chunksize: int,
    workers: int,
) -> Tuple[str, bool]:
    _ = chunksize
    started_at = time.monotonic()
    output_path = Path(output_parquet)
    schema_path = Path(schema_json)
    meta_path = Path(meta_json)

    if output_path.exists() and not force_rebuild:
        return (
            f"Feature parquet already exists at {output_parquet}; reusing (pass --force-rebuild to regenerate).",
            False,
        )

    cfg = _load_generator_config(window_config_yaml)
    window_values = _parse_window_values(cfg)
    worker_count = _worker_count(workers)

    if sys.platform != "linux" and worker_count > 1:
        worker_count = 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    _log("Phase 1/4: load input and split by symbol in memory")
    raw = _prepare_input_df(input_csv)
    universe = sorted(raw["symbol"].unique().tolist())
    symbol_frames = {symbol: g.reset_index(drop=True) for symbol, g in raw.groupby("symbol", sort=False)}
    _log(f"Universe size: {len(universe)} symbols")

    _log("Phase 2/4: build global context maps")
    ret_sum_global: Dict[pd.Timestamp, float] = {}
    ret_count_global: Dict[pd.Timestamp, int] = {}
    accel_ctx_global = _empty_accel_ctx(window_values["accel_roc_window"], window_values["accel_shift_window"])

    mp_context = mp.get_context("fork") if sys.platform == "linux" else None
    if worker_count == 1:
        for symbol in universe:
            ret_sum, ret_count, accel_ctx = _symbol_context_partials(
                symbol_frames[symbol], window_values["accel_roc_window"], window_values["accel_shift_window"]
            )
            _merge_sum_count_maps(ret_sum_global, ret_count_global, ret_sum, ret_count)
            for key, (sum_map, count_map) in accel_ctx.items():
                dst_sum, dst_count = accel_ctx_global[key]
                _merge_sum_count_maps(dst_sum, dst_count, sum_map, count_map)
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp_context,
            initializer=_init_context_worker,
            initargs=(symbol_frames, window_values["accel_roc_window"], window_values["accel_shift_window"]),
        ) as executor:
            futures = [executor.submit(_context_worker, symbol) for symbol in universe]
            for future in as_completed(futures):
                ret_sum, ret_count, accel_ctx = future.result()
                _merge_sum_count_maps(ret_sum_global, ret_count_global, ret_sum, ret_count)
                for key, (sum_map, count_map) in accel_ctx.items():
                    dst_sum, dst_count = accel_ctx_global[key]
                    _merge_sum_count_maps(dst_sum, dst_count, sum_map, count_map)

    _log("Phase 3/4: compute wide features and stream-write parquet")
    total_rows = 0
    generated_symbols = 0
    date_min_global: Optional[str] = None
    date_max_global: Optional[str] = None
    sample_df = pd.DataFrame()
    writer: Optional[pq.ParquetWriter] = None

    def _consume_result(symbol: str, wide: pd.DataFrame, date_min: str, date_max: str) -> None:
        nonlocal total_rows, generated_symbols, date_min_global, date_max_global, sample_df, writer
        if sample_df.empty:
            sample_df = wide.head(500)
        total_rows += int(len(wide))
        generated_symbols += 1
        date_min_global = date_min if date_min_global is None else min(date_min_global, date_min)
        date_max_global = date_max if date_max_global is None else max(date_max_global, date_max)

        table = pa.Table.from_pandas(wide, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
        writer.write_table(table)

    if worker_count == 1:
        for symbol in universe:
            symbol_name, wide, dmin, dmax = _wide_worker_local(
                symbol, symbol_frames, (ret_sum_global, ret_count_global), accel_ctx_global, cfg, window_values
            )
            _consume_result(symbol_name, wide, dmin, dmax)
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp_context,
            initializer=_init_wide_worker,
            initargs=(symbol_frames, (ret_sum_global, ret_count_global), accel_ctx_global, cfg, window_values),
        ) as executor:
            futures = [executor.submit(_wide_worker, symbol) for symbol in universe]
            for future in as_completed(futures):
                symbol_name, wide, dmin, dmax = future.result()
                _consume_result(symbol_name, wide, dmin, dmax)

    if writer is not None:
        writer.close()

    _log("Phase 4/4: write schema and metadata")
    _write_schema(schema_path, sample_df)
    _write_meta(
        meta_path,
        total_rows=total_rows,
        symbol_count=generated_symbols,
        date_min=date_min_global,
        date_max=date_max_global,
        universe=universe,
        cfg=cfg,
        workers=worker_count,
    )

    elapsed = time.monotonic() - started_at
    return (
        f"Built wide Parquet features for {generated_symbols} symbols with {total_rows} rows at {output_parquet} in {elapsed:.1f}s",
        True,
    )


def _wide_worker_local(
    symbol: str,
    symbol_frames: Dict[str, pd.DataFrame],
    ret_ctx: Tuple[Dict[pd.Timestamp, float], Dict[pd.Timestamp, int]],
    accel_ctx_map: Dict[Tuple[int, int], Tuple[Dict[pd.Timestamp, float], Dict[pd.Timestamp, int]]],
    cfg: Dict[str, object],
    window_values: Dict[str, List[int]],
) -> Tuple[str, pd.DataFrame, str, str]:
    smoothing = cfg["smoothing"]
    wide = _compute_symbol_wide(
        sdf=symbol_frames[symbol],
        ret_ctx=ret_ctx,
        accel_ctx_map=accel_ctx_map,
        window_values=window_values,
        accel_ema_window=int(smoothing["accel_ema_window"]),
        vol_ema_window=int(smoothing["vol_ema_window"]),
        price_momentum_source=str(cfg["price_momentum_source"]),
    )
    return symbol, wide, str(wide["date"].min().date()), str(wide["date"].max().date())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build option strategy feature store (wide parquet).")
    parser.add_argument("--input-csv", default="data/etfs.csv")
    parser.add_argument("--output-parquet", default="data/features/option_strategy_features.parquet")
    parser.add_argument("--schema-json", default="data/option_strategy_features.schema.json")
    parser.add_argument("--meta-json", default="data/option_strategy_features.meta.json")
    parser.add_argument("--window-config-yaml", default="data/option_feature_windows.yaml")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    message, _ = build_features(
        input_csv=args.input_csv,
        output_parquet=args.output_parquet,
        schema_json=args.schema_json,
        meta_json=args.meta_json,
        window_config_yaml=args.window_config_yaml,
        force_rebuild=args.force_rebuild,
        chunksize=args.chunksize,
        workers=args.workers,
    )
    print(message)


if __name__ == "__main__":
    main()
