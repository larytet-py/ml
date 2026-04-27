#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from option_pricing import BlackScholesPricer
from weekly_option_backtest_common import summarize_trades


PRICING_VOL_FALLBACK = 0.20
REQUIRED_METRIC_KEYS = [
    "total",
    "wins",
    "win_rate",
    "itm_expiries",
    "itm_rate",
    "total_pnl",
    "avg_pnl",
    "median_pnl",
    "avg_return_on_spot",
    "max_drawdown",
]
TRADE_COLUMNS = [
    "side",
    "entry_date",
    "exit_date",
    "scheduled_expiry_date",
    "time_to_expiry_days",
    "entry_close",
    "exit_close",
    "strike",
    "premium",
    "intrinsic_at_expiry",
    "expired_itm",
    "pnl_per_share",
    "pnl_per_contract",
    "roc_signal",
    "trend_vol_signal",
    "pricing_vol",
]


@dataclass
class StrategyKnobs:
    side: str
    roc_window_size: int
    roc_comparator: str
    roc_threshold: float
    roc_range_enabled: int
    roc_range_low: float
    roc_range_high: float
    vol_window_size: int
    vol_comparator: str
    vol_threshold: float
    vol_range_enabled: int
    vol_range_low: float
    vol_range_high: float


@dataclass
class BacktestEvaluation:
    feasible: bool
    infeasible_reason: Optional[str]
    metrics: Dict[str, float]
    trades_df: pd.DataFrame
    resolved_knobs: Dict[str, float]


def _empty_metrics() -> Dict[str, float]:
    return {
        "total": 0.0,
        "wins": 0.0,
        "win_rate": 0.0,
        "itm_expiries": 0.0,
        "itm_rate": 0.0,
        "total_pnl": 0.0,
        "avg_pnl": 0.0,
        "median_pnl": 0.0,
        "avg_return_on_spot": 0.0,
        "max_drawdown": 0.0,
    }


def _ensure_trade_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TRADE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[TRADE_COLUMNS]


def _safe_feature_bounds(values: pd.Series) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce")
    lo = float(np.nanmin(finite.to_numpy(dtype=float)))
    hi = float(np.nanmax(finite.to_numpy(dtype=float)))
    return lo, hi


def _build_signal_column_names(knobs: StrategyKnobs) -> tuple[str, str]:
    roc_col = f"roc_close_w{int(knobs.roc_window_size)}"
    if knobs.side == "put":
        vol_col = f"downside_vol_w{int(knobs.vol_window_size)}"
    else:
        vol_col = f"upside_vol_w{int(knobs.vol_window_size)}"
    return roc_col, vol_col


def _resolve_pricing_vol_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("realized_vol_close_w21", "realized_vol_close_w20", "realized_vol_close_w22"):
        if col in df.columns:
            return col
    return None


def _apply_rule(
    values: np.ndarray,
    comparator: str,
    threshold: float,
    range_enabled: int,
    range_low: float,
    range_high: float,
) -> np.ndarray:
    if int(range_enabled) == 1:
        return (values >= float(range_low)) & (values <= float(range_high))
    if comparator == "above":
        return values >= float(threshold)
    return values <= float(threshold)


def _coerce_knobs(raw: Dict[str, object]) -> StrategyKnobs:
    return StrategyKnobs(
        side=str(raw.get("side", "put")).strip().lower(),
        roc_window_size=int(round(float(raw.get("roc_window_size", 2)))),
        roc_comparator=str(raw.get("roc_comparator", "below")).strip().lower(),
        roc_threshold=float(raw.get("roc_threshold", 0.0)),
        roc_range_enabled=int(round(float(raw.get("roc_range_enabled", 0)))),
        roc_range_low=float(raw.get("roc_range_low", 0.0)),
        roc_range_high=float(raw.get("roc_range_high", 0.0)),
        vol_window_size=int(round(float(raw.get("vol_window_size", 2)))),
        vol_comparator=str(raw.get("vol_comparator", "above")).strip().lower(),
        vol_threshold=float(raw.get("vol_threshold", 0.0)),
        vol_range_enabled=int(round(float(raw.get("vol_range_enabled", 0)))),
        vol_range_low=float(raw.get("vol_range_low", 0.0)),
        vol_range_high=float(raw.get("vol_range_high", 0.0)),
    )


class PrecomputedFeatureBacktester:
    def __init__(self, features_df: pd.DataFrame, feature_data_version: str = "unknown") -> None:
        if "symbol" not in features_df.columns or "date" not in features_df.columns:
            raise ValueError("Expected precomputed features to include 'symbol' and 'date'.")
        required_price_cols = {"open", "high", "low", "close"}
        missing_price = sorted(c for c in required_price_cols if c not in features_df.columns)
        if missing_price:
            raise ValueError(f"Expected precomputed features to include price columns: missing {missing_price}")

        out = features_df.copy()
        out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        for col in ("open", "high", "low", "close", "volume"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        self.df = out.dropna(subset=["symbol", "date", "open", "high", "low", "close"]).sort_values(
            ["symbol", "date"]
        )
        self.feature_data_version = str(feature_data_version)

    @classmethod
    def from_parquet(cls, path: str) -> "PrecomputedFeatureBacktester":
        p = Path(path)
        df = pd.read_parquet(p)
        version = "unknown"
        meta_path = p.with_suffix(".meta.json")
        if meta_path.exists():
            loaded = json.loads(meta_path.read_text())
            version = str(loaded.get("feature_version", "unknown"))
        return cls(features_df=df, feature_data_version=version)

    def _slice_symbol(self, symbol: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        sdf = self.df[self.df["symbol"] == symbol.upper()].copy()
        if start_date:
            sdf = sdf[sdf["date"] >= pd.to_datetime(start_date)]
        if end_date:
            sdf = sdf[sdf["date"] <= pd.to_datetime(end_date)]
        return sdf.sort_values("date").reset_index(drop=True)

    def evaluate(
        self,
        *,
        knobs_input: Dict[str, object],
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        risk_free_rate: float = 0.04,
        min_pricing_vol_annualized: float = 0.10,
        contract_size: int = 100,
    ) -> BacktestEvaluation:
        knobs = _coerce_knobs(knobs_input)
        if knobs.side not in {"put", "call"}:
            return BacktestEvaluation(
                feasible=False,
                infeasible_reason=f"Invalid side '{knobs.side}' (expected 'put' or 'call').",
                metrics=_empty_metrics(),
                trades_df=_ensure_trade_columns(pd.DataFrame()),
                resolved_knobs=asdict(knobs),
            )
        if knobs.roc_comparator not in {"above", "below"}:
            return BacktestEvaluation(
                feasible=False,
                infeasible_reason=f"Invalid roc comparator '{knobs.roc_comparator}'.",
                metrics=_empty_metrics(),
                trades_df=_ensure_trade_columns(pd.DataFrame()),
                resolved_knobs=asdict(knobs),
            )
        if knobs.vol_comparator not in {"above", "below"}:
            return BacktestEvaluation(
                feasible=False,
                infeasible_reason=f"Invalid vol comparator '{knobs.vol_comparator}'.",
                metrics=_empty_metrics(),
                trades_df=_ensure_trade_columns(pd.DataFrame()),
                resolved_knobs=asdict(knobs),
            )
        if knobs.roc_range_enabled == 1 and knobs.roc_range_low > knobs.roc_range_high:
            return BacktestEvaluation(
                feasible=False,
                infeasible_reason="Invalid ROC range: roc_range_low > roc_range_high.",
                metrics=_empty_metrics(),
                trades_df=_ensure_trade_columns(pd.DataFrame()),
                resolved_knobs=asdict(knobs),
            )
        if knobs.vol_range_enabled == 1 and knobs.vol_range_low > knobs.vol_range_high:
            return BacktestEvaluation(
                feasible=False,
                infeasible_reason="Invalid vol range: vol_range_low > vol_range_high.",
                metrics=_empty_metrics(),
                trades_df=_ensure_trade_columns(pd.DataFrame()),
                resolved_knobs=asdict(knobs),
            )

        sdf = self._slice_symbol(symbol=symbol, start_date=start_date, end_date=end_date)
        if sdf.empty:
            return BacktestEvaluation(
                feasible=False,
                infeasible_reason=f"No rows for symbol={symbol.upper()} in requested date range.",
                metrics=_empty_metrics(),
                trades_df=_ensure_trade_columns(pd.DataFrame()),
                resolved_knobs=asdict(knobs),
            )

        roc_col, vol_col = _build_signal_column_names(knobs)
        missing_cols = [c for c in (roc_col, vol_col) if c not in sdf.columns]
        if missing_cols:
            return BacktestEvaluation(
                feasible=False,
                infeasible_reason=f"Missing precomputed signal columns: {missing_cols}",
                metrics=_empty_metrics(),
                trades_df=_ensure_trade_columns(pd.DataFrame()),
                resolved_knobs=asdict(knobs),
            )

        pricing_vol_col = _resolve_pricing_vol_column(sdf)
        if pricing_vol_col is None:
            sdf["__pricing_vol"] = PRICING_VOL_FALLBACK
            pricing_vol_col = "__pricing_vol"

        roc_lo, roc_hi = _safe_feature_bounds(sdf[roc_col])
        vol_lo, vol_hi = _safe_feature_bounds(sdf[vol_col])
        if knobs.roc_range_enabled == 1:
            knobs.roc_range_low = max(roc_lo, min(roc_hi, knobs.roc_range_low))
            knobs.roc_range_high = max(roc_lo, min(roc_hi, knobs.roc_range_high))
        if knobs.vol_range_enabled == 1:
            knobs.vol_range_low = max(vol_lo, min(vol_hi, knobs.vol_range_low))
            knobs.vol_range_high = max(vol_lo, min(vol_hi, knobs.vol_range_high))

        pricer = BlackScholesPricer(risk_free_rate=risk_free_rate, min_sigma=min_pricing_vol_annualized)

        dates = sdf["date"]
        weekday = dates.dt.weekday.to_numpy(dtype=int, copy=False)
        days_to_friday = 4 - weekday
        week_key = dates.dt.isocalendar()["year"].astype(int) * 100 + dates.dt.isocalendar()["week"].astype(int)
        week_last_idx = (
            pd.Series(sdf.index, index=sdf.index).groupby(week_key).transform("max").to_numpy(dtype=int, copy=False)
        )

        roc = pd.to_numeric(sdf[roc_col], errors="coerce").to_numpy(dtype=float, copy=False)
        trend_vol = pd.to_numeric(sdf[vol_col], errors="coerce").to_numpy(dtype=float, copy=False)
        pricing_vol_raw = pd.to_numeric(sdf[pricing_vol_col], errors="coerce").to_numpy(dtype=float, copy=False)

        base_mask = (~np.isnan(roc)) & (~np.isnan(trend_vol)) & (~np.isnan(pricing_vol_raw)) & (days_to_friday > 0)
        roc_trigger = _apply_rule(
            values=roc,
            comparator=knobs.roc_comparator,
            threshold=knobs.roc_threshold,
            range_enabled=knobs.roc_range_enabled,
            range_low=knobs.roc_range_low,
            range_high=knobs.roc_range_high,
        )
        vol_trigger = _apply_rule(
            values=trend_vol,
            comparator=knobs.vol_comparator,
            threshold=knobs.vol_threshold,
            range_enabled=knobs.vol_range_enabled,
            range_low=knobs.vol_range_low,
            range_high=knobs.vol_range_high,
        )
        trigger_mask = base_mask & roc_trigger & vol_trigger

        close = pd.to_numeric(sdf["close"], errors="coerce").to_numpy(dtype=float, copy=False)
        all_dates = dates.to_numpy()
        next_entry_idx = 0
        trades = []
        for i in np.flatnonzero(trigger_mask):
            if i < next_entry_idx:
                continue
            exit_idx = int(week_last_idx[i])
            if exit_idx >= len(sdf) or exit_idx <= i:
                continue
            d2f = int(days_to_friday[i])
            entry_close = float(close[i])
            entry_date = pd.Timestamp(all_dates[i])
            strike = entry_close
            time_to_expiry_days = d2f
            time_to_expiry_years = float(time_to_expiry_days) / 365.25
            scheduled_expiry_date = (entry_date + pd.Timedelta(days=d2f)).normalize()

            raw_sigma = float(pricing_vol_raw[i])
            used_sigma = pricer.effective_sigma(raw_sigma)
            premium = pricer.price(
                side=knobs.side,
                spot=entry_close,
                strike=strike,
                time_to_expiry_years=time_to_expiry_years,
                sigma=raw_sigma,
            )

            exit_close = float(close[exit_idx])
            intrinsic = pricer.intrinsic_value(side=knobs.side, strike=strike, spot=exit_close)
            expired_itm = intrinsic > 0.0
            pnl_per_share = premium - intrinsic

            trades.append(
                {
                    "side": knobs.side,
                    "entry_date": pd.Timestamp(all_dates[i]),
                    "exit_date": pd.Timestamp(all_dates[exit_idx]),
                    "scheduled_expiry_date": scheduled_expiry_date,
                    "time_to_expiry_days": time_to_expiry_days,
                    "entry_close": entry_close,
                    "exit_close": exit_close,
                    "strike": strike,
                    "premium": premium,
                    "intrinsic_at_expiry": intrinsic,
                    "expired_itm": expired_itm,
                    "pnl_per_share": pnl_per_share,
                    "pnl_per_contract": pnl_per_share * contract_size,
                    "roc_signal": float(roc[i]),
                    "trend_vol_signal": float(trend_vol[i]),
                    "pricing_vol": float(used_sigma),
                }
            )
            next_entry_idx = exit_idx + 1

        trades_df = _ensure_trade_columns(pd.DataFrame(trades))
        metrics = summarize_trades(trades_df)
        metrics = _empty_metrics() if metrics is None else {k: float(metrics.get(k, 0.0)) for k in REQUIRED_METRIC_KEYS}
        return BacktestEvaluation(
            feasible=True,
            infeasible_reason=None,
            metrics=metrics,
            trades_df=trades_df,
            resolved_knobs=asdict(knobs),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one option-strategy knob set on precomputed feature parquet data. "
            "Designed for Sobol/gradient workflows where features are built once and reused "
            "across many backtest evaluations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example (emulate reversal-style put rule):\n"
            "  python3 backtest_option_strategy_sobol_gradient.py "
            "--symbol VXX --side put --roc-window-size 11 --roc-comparator below "
            "--roc-threshold -0.081758 --vol-window-size 21 --vol-comparator above "
            "--vol-threshold 0.379078"
        ),
    )
    parser.add_argument(
        "--features-parquet",
        default="data/features/option_strategy_features.parquet",
        help="Path to precomputed features parquet produced by build_option_strategy_features.py.",
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Ticker/symbol to backtest (case-insensitive).",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Inclusive start date filter (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive end date filter (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--side",
        choices=["put", "call"],
        required=True,
        help="Option side to short at entry.",
    )
    parser.add_argument(
        "--roc-window-size",
        type=int,
        required=True,
        help="ROC lookback window used to select feature column roc_close_w{window}.",
    )
    parser.add_argument(
        "--roc-comparator",
        choices=["above", "below"],
        default="below",
        help="Comparator for ROC threshold rule when --roc-range-enabled=0.",
    )
    parser.add_argument(
        "--roc-threshold",
        type=float,
        default=0.0,
        help="ROC threshold used when --roc-range-enabled=0.",
    )
    parser.add_argument(
        "--roc-range-enabled",
        type=int,
        default=0,
        help="Set to 1 to use ROC interval trigger [low, high] and ignore comparator/threshold; 0 to disable.",
    )
    parser.add_argument(
        "--roc-range-low",
        type=float,
        default=0.0,
        help="Lower bound for ROC interval trigger when --roc-range-enabled=1.",
    )
    parser.add_argument(
        "--roc-range-high",
        type=float,
        default=0.0,
        help="Upper bound for ROC interval trigger when --roc-range-enabled=1.",
    )
    parser.add_argument(
        "--vol-window-size",
        type=int,
        required=True,
        help="Volatility window used to select trend-vol feature column (downside_vol_w* for put, upside_vol_w* for call).",
    )
    parser.add_argument(
        "--vol-comparator",
        choices=["above", "below"],
        default="above",
        help="Comparator for vol threshold rule when --vol-range-enabled=0.",
    )
    parser.add_argument(
        "--vol-threshold",
        type=float,
        default=0.0,
        help="Trend-vol threshold used when --vol-range-enabled=0.",
    )
    parser.add_argument(
        "--vol-range-enabled",
        type=int,
        default=0,
        help="Set to 1 to use vol interval trigger [low, high] and ignore comparator/threshold; 0 to disable.",
    )
    parser.add_argument(
        "--vol-range-low",
        type=float,
        default=0.0,
        help="Lower bound for vol interval trigger when --vol-range-enabled=1.",
    )
    parser.add_argument(
        "--vol-range-high",
        type=float,
        default=0.0,
        help="Upper bound for vol interval trigger when --vol-range-enabled=1.",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.04,
        help="Annual risk-free rate passed into Black-Scholes pricing.",
    )
    parser.add_argument(
        "--min-pricing-vol",
        type=float,
        default=0.10,
        help="Minimum annualized sigma floor used in pricing (effective_sigma=max(raw_sigma, floor)).",
    )
    parser.add_argument(
        "--contract-size",
        type=int,
        default=100,
        help="Contract multiplier used to compute pnl_per_contract.",
    )
    parser.add_argument(
        "--trades-parquet",
        default=None,
        help="Optional output parquet path for full trade-level rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backtester = PrecomputedFeatureBacktester.from_parquet(args.features_parquet)
    result = backtester.evaluate(
        knobs_input={
            "side": args.side,
            "roc_window_size": args.roc_window_size,
            "roc_comparator": args.roc_comparator,
            "roc_threshold": args.roc_threshold,
            "roc_range_enabled": args.roc_range_enabled,
            "roc_range_low": args.roc_range_low,
            "roc_range_high": args.roc_range_high,
            "vol_window_size": args.vol_window_size,
            "vol_comparator": args.vol_comparator,
            "vol_threshold": args.vol_threshold,
            "vol_range_enabled": args.vol_range_enabled,
            "vol_range_low": args.vol_range_low,
            "vol_range_high": args.vol_range_high,
        },
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        risk_free_rate=args.risk_free_rate,
        min_pricing_vol_annualized=args.min_pricing_vol,
        contract_size=args.contract_size,
    )

    if args.trades_parquet:
        out_path = Path(args.trades_parquet)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.trades_df.to_parquet(out_path, index=False)

    print(
        json.dumps(
            {
                "feasible": bool(result.feasible),
                "infeasible_reason": result.infeasible_reason,
                "metrics": result.metrics,
                "trade_count": int(len(result.trades_df)),
                "resolved_knobs": result.resolved_knobs,
                "feature_data_version": backtester.feature_data_version,
                "trades_parquet": args.trades_parquet,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
