import unittest

import pandas as pd

from backtest_option_strategy_sobol_gradient import PrecomputedFeatureBacktester


def _sample_feature_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=10, freq="B")
    close = [100.0, 101.0, 100.5, 101.5, 102.0, 101.0, 100.8, 101.2, 101.6, 102.1]
    return pd.DataFrame(
        {
            "symbol": ["SPY"] * len(dates),
            "date": dates,
            "open": close,
            "high": [x + 1.0 for x in close],
            "low": [x - 1.0 for x in close],
            "close": close,
            "volume": [1_000_000] * len(dates),
            "roc_close_w2": [0.20, 0.22, 0.24, 0.18, 0.19, 0.21, 0.11, 0.09, 0.12, 0.13],
            "downside_vol_w2": [0.40, 0.41, 0.42, 0.39, 0.40, 0.38, 0.37, 0.36, 0.35, 0.34],
            "upside_vol_w2": [0.20, 0.21, 0.22, 0.19, 0.20, 0.18, 0.17, 0.16, 0.15, 0.14],
            "realized_vol_close_w21": [0.25] * len(dates),
        }
    )


class SobolGradientBacktestTests(unittest.TestCase):
    GOLDEN_EXPECTED = {
        "feature_data_version": "unknown",
        "trade_count": 2,
        "metrics": {
            "avg_pnl": 95.81020969175462,
            "avg_return_on_spot": 0.009545525256102892,
            "itm_expiries": 0.0,
            "itm_rate": 0.0,
            "max_drawdown": 0.0,
            "median_pnl": 95.81020969175462,
            "total": 2.0,
            "total_pnl": 191.62041938350924,
            "win_rate": 1.0,
            "wins": 2.0,
        },
        "resolved_knobs": {
            "roc_comparator": "below",
            "roc_range_enabled": 0,
            "roc_range_high": 0.0,
            "roc_range_low": 0.0,
            "roc_threshold": 0.2,
            "roc_window_size": 2,
            "side": "put",
            "vol_comparator": "above",
            "vol_range_enabled": 0,
            "vol_range_high": 0.0,
            "vol_range_low": 0.0,
            "vol_threshold": 0.35,
            "vol_window_size": 2,
        },
    }

    def test_range_precedence_over_comparator(self):
        backtester = PrecomputedFeatureBacktester(_sample_feature_frame())
        result = backtester.evaluate(
            knobs_input={
                "side": "put",
                "roc_window_size": 2,
                "roc_comparator": "below",
                "roc_threshold": -0.5,
                "roc_range_enabled": 1,
                "roc_range_low": 0.1,
                "roc_range_high": 0.3,
                "vol_window_size": 2,
                "vol_comparator": "above",
                "vol_threshold": 0.2,
                "vol_range_enabled": 0,
                "vol_range_low": 0.0,
                "vol_range_high": 0.0,
            },
            symbol="SPY",
        )
        self.assertTrue(result.feasible)
        self.assertGreater(len(result.trades_df), 0)

    def test_invalid_range_is_infeasible(self):
        backtester = PrecomputedFeatureBacktester(_sample_feature_frame())
        result = backtester.evaluate(
            knobs_input={
                "side": "put",
                "roc_window_size": 2,
                "roc_comparator": "below",
                "roc_threshold": 0.0,
                "roc_range_enabled": 1,
                "roc_range_low": 0.3,
                "roc_range_high": 0.1,
                "vol_window_size": 2,
                "vol_comparator": "above",
                "vol_threshold": 0.2,
                "vol_range_enabled": 0,
                "vol_range_low": 0.0,
                "vol_range_high": 0.0,
            },
            symbol="SPY",
        )
        self.assertFalse(result.feasible)
        self.assertIn("roc_range_low > roc_range_high", str(result.infeasible_reason))

    def test_required_trade_and_metric_fields(self):
        backtester = PrecomputedFeatureBacktester(_sample_feature_frame())
        result = backtester.evaluate(
            knobs_input={
                "side": "call",
                "roc_window_size": 2,
                "roc_comparator": "above",
                "roc_threshold": 0.1,
                "roc_range_enabled": 0,
                "roc_range_low": 0.0,
                "roc_range_high": 0.0,
                "vol_window_size": 2,
                "vol_comparator": "above",
                "vol_threshold": 0.1,
                "vol_range_enabled": 0,
                "vol_range_low": 0.0,
                "vol_range_high": 0.0,
            },
            symbol="SPY",
        )
        self.assertTrue(result.feasible)
        for key in [
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
        ]:
            self.assertIn(key, result.metrics)

        for col in [
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
        ]:
            self.assertIn(col, result.trades_df.columns)

    def test_sample_fixture_matches_golden_output(self):
        backtester = PrecomputedFeatureBacktester(_sample_feature_frame())
        result = backtester.evaluate(
            knobs_input={
                "side": "put",
                "roc_window_size": 2,
                "roc_comparator": "below",
                "roc_threshold": 0.2,
                "roc_range_enabled": 0,
                "roc_range_low": 0.0,
                "roc_range_high": 0.0,
                "vol_window_size": 2,
                "vol_comparator": "above",
                "vol_threshold": 0.35,
                "vol_range_enabled": 0,
                "vol_range_low": 0.0,
                "vol_range_high": 0.0,
            },
            symbol="SPY",
        )

        self.assertTrue(result.feasible)
        self.assertEqual(backtester.feature_data_version, self.GOLDEN_EXPECTED["feature_data_version"])
        self.assertEqual(len(result.trades_df), self.GOLDEN_EXPECTED["trade_count"])
        self.assertEqual(result.resolved_knobs, self.GOLDEN_EXPECTED["resolved_knobs"])
        for key, expected_value in self.GOLDEN_EXPECTED["metrics"].items():
            self.assertAlmostEqual(result.metrics[key], expected_value, places=12, msg=key)


if __name__ == "__main__":
    unittest.main()
