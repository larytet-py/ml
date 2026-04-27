import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from build_option_strategy_features import build_features


TRADING_DAYS_PER_YEAR = 252
EPS = 1e-12

# Mirrors the feature groups defined in sobol_gradient_descent_optimization_plan.md.
PLAN_FEATURE_GROUPS = [
    "ret_1d_close",
    "ret_1d_open",
    "roc_close_window",
    "roc_open_window",
    "accel_close_window_shift",
    "accel_open_window_shift",
    "accel_close_ema_window",
    "accel_open_ema_window",
    "accel_regime_sign_window",
    "accel_corr_market_window",
    "realized_vol_close_window",
    "realized_vol_open_window",
    "downside_vol_window",
    "upside_vol_window",
    "realized_vol_close_ema_window",
    "realized_vol_open_ema_window",
    "downside_vol_ema_window",
    "upside_vol_ema_window",
    "price_momentum_window",
    "price_stddev_window",
    "volume_momentum_window",
    "volume_roc_window",
    "volume_stddev_window",
    "day_range_over_open",
    "close_position_in_range",
    "body_over_open",
    "corr_market_window",
]


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


def _prepare_independent_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["symbol", "date"]).drop_duplicates(subset=["symbol", "date"], keep="last").reset_index(drop=True)
    return out[["symbol", "date", "open", "high", "low", "close", "volume"]]


def _sum_count_by_date(dates: pd.Series, values: pd.Series) -> tuple[dict[pd.Timestamp, float], dict[pd.Timestamp, int]]:
    valid = ~values.isna()
    tmp = pd.DataFrame({"date": dates[valid], "value": values[valid]})
    if tmp.empty:
        return {}, {}
    grouped = tmp.groupby("date")["value"]
    return grouped.sum().to_dict(), grouped.count().astype(int).to_dict()


def _build_independent_context(
    prepared: pd.DataFrame,
    accel_roc_values: list[int],
    accel_shift_values: list[int],
) -> tuple[
    tuple[dict[pd.Timestamp, float], dict[pd.Timestamp, int]],
    dict[tuple[int, int], tuple[dict[pd.Timestamp, float], dict[pd.Timestamp, int]]],
]:
    per_symbol = prepared.copy()
    per_symbol["ret_1d_close"] = per_symbol.groupby("symbol", sort=False)["close"].pct_change(1)
    ret_ctx = _sum_count_by_date(per_symbol["date"], per_symbol["ret_1d_close"])

    accel_ctx_map: dict[tuple[int, int], tuple[dict[pd.Timestamp, float], dict[pd.Timestamp, int]]] = {}
    by_symbol_close = per_symbol.groupby("symbol", sort=False)["close"]
    for ar in accel_roc_values:
        close_roc = by_symbol_close.pct_change(ar)
        for sh in accel_shift_values:
            accel = close_roc - close_roc.groupby(per_symbol["symbol"], sort=False).shift(sh)
            accel_ctx_map[(ar, sh)] = _sum_count_by_date(per_symbol["date"], accel)
    return ret_ctx, accel_ctx_map


def _compute_independent_symbol_features(
    sdf: pd.DataFrame,
    ret_ctx: tuple[dict[pd.Timestamp, float], dict[pd.Timestamp, int]],
    accel_ctx_map: dict[tuple[int, int], tuple[dict[pd.Timestamp, float], dict[pd.Timestamp, int]]],
    *,
    roc_values: list[int],
    accel_roc_values: list[int],
    accel_shift_values: list[int],
    vol_values: list[int],
    corr_values: list[int],
    accel_ema_window: int,
    vol_ema_window: int,
    price_momentum_source: str,
) -> pd.DataFrame:
    out = sdf.copy().reset_index(drop=True)
    feature_map: dict[str, pd.Series] = {}

    ret_1d_close = out["close"].pct_change(1)
    ret_1d_open = out["open"].pct_change(1)
    ret_sum_map, ret_count_map = ret_ctx
    ret_sum = out["date"].map(ret_sum_map)
    ret_count = out["date"].map(ret_count_map)
    denom_ret = (ret_count - 1).where((ret_count - 1) > 0, np.nan)
    market_ex_self_ret = (ret_sum - ret_1d_close) / denom_ret

    feature_map["ret_1d_close"] = ret_1d_close
    feature_map["ret_1d_open"] = ret_1d_open
    feature_map["day_range_over_open"] = (out["high"] - out["low"]) / out["open"]
    feature_map["close_position_in_range"] = (out["close"] - out["low"]) / ((out["high"] - out["low"]) + EPS)
    feature_map["body_over_open"] = (out["close"] - out["open"]) / out["open"]

    for w in roc_values:
        vol_mom = (out["volume"].shift(1) / out["volume"].shift(1 + w)) - 1.0
        feature_map[f"roc_close_w{w}"] = out["close"].pct_change(w)
        feature_map[f"roc_open_w{w}"] = out["open"].pct_change(w)
        feature_map[f"price_momentum_w{w}"] = out[price_momentum_source].pct_change(w)
        feature_map[f"volume_momentum_w{w}"] = vol_mom
        feature_map[f"volume_roc_w{w}"] = vol_mom

    for ar in accel_roc_values:
        close_roc = out["close"].pct_change(ar)
        open_roc = out["open"].pct_change(ar)
        for sh in accel_shift_values:
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
            for cw in corr_values:
                feature_map[f"accel_corr_market_ar{ar}_sh{sh}_cw{cw}"] = accel_close.rolling(cw).corr(market_ex_self_accel)

    for vw in vol_values:
        rv_close = ret_1d_close.rolling(vw).std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)
        rv_open = ret_1d_open.rolling(vw).std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)
        dvol = ret_1d_close.rolling(vw).apply(_downside_std, raw=True) * math.sqrt(TRADING_DAYS_PER_YEAR)
        uvol = ret_1d_close.rolling(vw).apply(_upside_std, raw=True) * math.sqrt(TRADING_DAYS_PER_YEAR)
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

    for cw in corr_values:
        feature_map[f"corr_market_cw{cw}"] = ret_1d_close.rolling(cw).corr(market_ex_self_ret)

    feat_df = pd.DataFrame(feature_map)
    return pd.concat([out[["symbol", "date"]], feat_df], axis=1)


class BuildOptionStrategyFeaturesHappyFlowTests(unittest.TestCase):
    def test_happy_flow_wide_parquet_two_symbols_short_windows(self):
        self.assertGreaterEqual(len(PLAN_FEATURE_GROUPS), 20)
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            input_csv = base / "etfs_small.csv"
            output_parquet = base / "features.parquet"
            schema_json = base / "features.schema.json"
            meta_json = base / "features.meta.json"
            window_yaml = base / "windows.yaml"

            source = pd.read_csv("data/etfs.csv")
            source["symbol"] = source["symbol"].astype(str).str.upper().str.strip()
            small = pd.concat(
                [
                    source[source["symbol"] == "SPY"].head(30),
                    source[source["symbol"] == "TLT"].head(30),
                ],
                ignore_index=True,
            )
            small.to_csv(input_csv, index=False)

            window_yaml.write_text(
                "\n".join(
                    [
                        "window_ranges:",
                        "  roc_window:",
                        "    min: 2",
                        "    max: 3",
                        "    step: 1",
                        "  accel_roc_window:",
                        "    min: 2",
                        "    max: 3",
                        "    step: 1",
                        "  accel_shift_window:",
                        "    min: 1",
                        "    max: 2",
                        "    step: 1",
                        "  vol_window:",
                        "    min: 2",
                        "    max: 3",
                        "    step: 1",
                        "  corr_window:",
                        "    min: 2",
                        "    max: 3",
                        "    step: 1",
                        "smoothing:",
                        "  accel_ema_window: 3",
                        "  vol_ema_window: 3",
                        "price_momentum_source: close",
                    ]
                )
            )

            msg, rebuilt = build_features(
                input_csv=str(input_csv),
                output_parquet=str(output_parquet),
                schema_json=str(schema_json),
                meta_json=str(meta_json),
                window_config_yaml=str(window_yaml),
                force_rebuild=True,
                chunksize=1000,
                workers=1,
            )

            self.assertTrue(rebuilt)
            self.assertIn("Built wide Parquet", msg)
            self.assertTrue(output_parquet.exists())
            self.assertTrue(schema_json.exists())
            self.assertTrue(meta_json.exists())

            feats = pd.read_parquet(output_parquet)
            self.assertEqual(sorted(feats["symbol"].unique().tolist()), ["SPY", "TLT"])
            self.assertGreater(len(feats), 0)

            expected_columns = [
                "roc_close_w2",
                "roc_open_w3",
                "accel_close_w2_s1",
                "accel_open_ema_w3_s2",
                "accel_corr_market_ar2_sh1_cw2",
                "realized_vol_close_w2",
                "downside_vol_w3",
                "realized_vol_open_ema_w2",
                "price_momentum_w2",
                "price_stddev_w2",
                "volume_momentum_w2",
                "volume_roc_w2",
                "volume_stddev_w3",
                "corr_market_cw2",
            ]

            for col in expected_columns:
                self.assertIn(col, feats.columns)

            for col in [
                "roc_close_w2",
                "accel_close_w2_s1",
                "accel_corr_market_ar2_sh1_cw2",
                "realized_vol_close_w2",
                "corr_market_cw2",
            ]:
                self.assertTrue(feats[col].notna().any(), msg=f"expected non-NaN values in {col}")

            prepared = _prepare_independent_df(small)
            ret_ctx, accel_ctx_map = _build_independent_context(prepared, accel_roc_values=[2, 3], accel_shift_values=[1, 2])

            symbols = sorted(prepared["symbol"].unique().tolist())
            self.assertEqual(symbols, ["SPY", "TLT"])

            compare_columns = [c for c in feats.columns if c not in {"symbol", "date", "open", "high", "low", "close", "volume"}]
            self.assertGreater(len(compare_columns), 0)

            feats_sorted = feats.sort_values(["symbol", "date"]).reset_index(drop=True)
            independent_frames: list[pd.DataFrame] = []
            for symbol in symbols:
                sdf = prepared[prepared["symbol"] == symbol].reset_index(drop=True)
                independent_frames.append(
                    _compute_independent_symbol_features(
                        sdf,
                        ret_ctx,
                        accel_ctx_map,
                        roc_values=[2, 3],
                        accel_roc_values=[2, 3],
                        accel_shift_values=[1, 2],
                        vol_values=[2, 3],
                        corr_values=[2, 3],
                        accel_ema_window=3,
                        vol_ema_window=3,
                        price_momentum_source="close",
                    )
                )
            independent = pd.concat(independent_frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)

            for col in compare_columns:
                assert_series_equal(
                    feats_sorted[col].astype(float),
                    independent[col].astype(float),
                    check_names=False,
                    rtol=1e-10,
                    atol=1e-12,
                )

            meta = json.loads(meta_json.read_text())
            self.assertEqual(meta["symbol_count"], 2)
            self.assertEqual(meta["output_format"], "parquet")


if __name__ == "__main__":
    unittest.main()
