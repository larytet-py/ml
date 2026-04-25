import unittest

from backtest_weekly_option_acceleration_reversal import run_backtest as run_accel_backtest
from backtest_weekly_option_acceleration_reversal import summarize_trades as summarize_accel_trades
from backtest_weekly_option_reversal import load_symbol_data
from backtest_weekly_option_reversal import run_backtest as run_roc_backtest
from backtest_weekly_option_reversal import summarize_trades as summarize_roc_trades
from backtest_weekly_option_roc_accel_reversal import run_backtest as run_roc_accel_backtest
from backtest_weekly_option_roc_accel_reversal import summarize_trades as summarize_roc_accel_trades


CSV_PATH = "data/etfs.csv"
RISK_FREE_RATE = 0.04
MIN_PRICING_VOL = 0.10
CONTRACT_SIZE = 100


class BacktestTableSamplesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spy_df = load_symbol_data(CSV_PATH, "SPY", None, None)

    def _assert_common_metrics(
        self,
        metrics,
        trades,
        expected_trades,
        expected_win_rate_pct,
        expected_itm_count,
        expected_itm_rate_pct,
        expected_total_pnl,
        expected_avg_pnl,
        expected_median_pnl,
        expected_avg_return_pct,
        expected_max_drawdown,
    ):
        self.assertIsNotNone(metrics)
        self.assertEqual(int(metrics["total"]), expected_trades)
        self.assertEqual(len(trades), expected_trades)
        self.assertEqual(round(metrics["win_rate"] * 100.0, 2), expected_win_rate_pct)
        self.assertEqual(int(metrics["itm_expiries"]), expected_itm_count)
        self.assertEqual(round(metrics["itm_rate"] * 100.0, 2), expected_itm_rate_pct)
        self.assertEqual(round(metrics["total_pnl"], 2), expected_total_pnl)
        self.assertEqual(round(metrics["avg_pnl"], 2), expected_avg_pnl)
        self.assertEqual(round(metrics["median_pnl"], 2), expected_median_pnl)
        self.assertEqual(round(metrics["avg_return_on_spot"] * 100.0, 4), expected_avg_return_pct)
        self.assertEqual(round(metrics["max_drawdown"], 2), expected_max_drawdown)

    def test_reversal_sample_from_config_matches_table(self):
        # option_signal_notifier.config sample:
        # --symbol SPY --side call --roc-lookback 22 --vol-window 2
        # --call-roc-threshold 0.000000 --upside-vol-threshold 0.019602
        trades = run_roc_backtest(
            df=self.spy_df.copy(),
            side="call",
            roc_lookback=22,
            put_roc_threshold=-0.03,
            call_roc_threshold=0.0,
            vol_window=2,
            downside_vol_threshold_annualized=0.2,
            upside_vol_threshold_annualized=0.019602,
            risk_free_rate=RISK_FREE_RATE,
            min_pricing_vol_annualized=MIN_PRICING_VOL,
            contract_size=CONTRACT_SIZE,
            allow_overlap=False,
        )
        metrics = summarize_roc_trades(trades)

        self._assert_common_metrics(
            metrics=metrics,
            trades=trades,
            expected_trades=22,
            expected_win_rate_pct=72.73,
            expected_itm_count=9,
            expected_itm_rate_pct=40.91,
            expected_total_pnl=1564.65,
            expected_avg_pnl=71.12,
            expected_median_pnl=136.82,
            expected_avg_return_pct=0.1056,
            expected_max_drawdown=-615.06,
        )

        expected_recent_entry_dates = [
            "2025-11-10",
            "2025-12-04",
            "2025-12-11",
            "2026-01-05",
            "2026-01-12",
            "2026-01-22",
            "2026-01-26",
            "2026-02-09",
            "2026-04-08",
            "2026-04-15",
        ]
        expected_recent_exit_dates = [
            "2025-11-14",
            "2025-12-05",
            "2025-12-12",
            "2026-01-09",
            "2026-01-16",
            "2026-01-23",
            "2026-01-30",
            "2026-02-13",
            "2026-04-10",
            "2026-04-17",
        ]
        self.assertEqual(
            [d.date().isoformat() for d in trades["entry_date"].tail(10)],
            expected_recent_entry_dates,
        )
        self.assertEqual(
            [d.date().isoformat() for d in trades["exit_date"].tail(10)],
            expected_recent_exit_dates,
        )

    def test_acceleration_sample_from_config_matches_table(self):
        # option_signal_notifier.config sample:
        # --symbol SPY --side call --accel-window 53 --vol-window 14
        # --call-accel-threshold -0.013715 --upside-vol-threshold 0.046678
        trades = run_accel_backtest(
            df=self.spy_df.copy(),
            side="call",
            accel_window=53,
            put_accel_threshold=-0.03,
            call_accel_threshold=-0.013715,
            vol_window=14,
            downside_vol_threshold_annualized=0.2,
            upside_vol_threshold_annualized=0.046678,
            risk_free_rate=RISK_FREE_RATE,
            min_pricing_vol_annualized=MIN_PRICING_VOL,
            contract_size=CONTRACT_SIZE,
            allow_overlap=False,
        )
        metrics = summarize_accel_trades(trades)

        self._assert_common_metrics(
            metrics=metrics,
            trades=trades,
            expected_trades=8,
            expected_win_rate_pct=87.50,
            expected_itm_count=2,
            expected_itm_rate_pct=25.00,
            expected_total_pnl=1106.86,
            expected_avg_pnl=138.36,
            expected_median_pnl=189.48,
            expected_avg_return_pct=0.2042,
            expected_max_drawdown=-445.48,
        )

        self.assertEqual(
            [d.date().isoformat() for d in trades["entry_date"]],
            [
                "2025-07-09",
                "2025-09-02",
                "2025-11-06",
                "2025-11-17",
                "2026-01-28",
                "2026-02-11",
                "2026-03-12",
                "2026-03-26",
            ],
        )
        self.assertEqual(
            [d.date().isoformat() for d in trades["exit_date"]],
            [
                "2025-07-11",
                "2025-09-05",
                "2025-11-07",
                "2025-11-21",
                "2026-01-30",
                "2026-02-13",
                "2026-03-13",
                "2026-03-27",
            ],
        )

    def test_roc_acceleration_sample_from_config_matches_table(self):
        # option_signal_notifier.config sample:
        # --symbol SPY --side put --roc-lookback 44 --accel-window 19
        # --put-roc-threshold -0.007364 --put-accel-threshold 0.008792
        trades = run_roc_accel_backtest(
            df=self.spy_df.copy(),
            side="put",
            roc_lookback=44,
            accel_window=19,
            put_roc_threshold=-0.007364,
            call_roc_threshold=0.03,
            put_accel_threshold=0.008792,
            call_accel_threshold=-0.03,
            risk_free_rate=RISK_FREE_RATE,
            min_pricing_vol_annualized=MIN_PRICING_VOL,
            contract_size=CONTRACT_SIZE,
            allow_overlap=False,
        )
        metrics = summarize_roc_accel_trades(trades)

        self._assert_common_metrics(
            metrics=metrics,
            trades=trades,
            expected_trades=4,
            expected_win_rate_pct=75.00,
            expected_itm_count=1,
            expected_itm_rate_pct=25.00,
            expected_total_pnl=-897.47,
            expected_avg_pnl=-224.37,
            expected_median_pnl=354.41,
            expected_avg_return_pct=-0.3489,
            expected_max_drawdown=-2023.69,
        )

        self.assertEqual(
            [d.date().isoformat() for d in trades["entry_date"]],
            ["2026-02-17", "2026-03-25", "2026-03-31", "2026-04-08"],
        )
        self.assertEqual(
            [d.date().isoformat() for d in trades["exit_date"]],
            ["2026-02-20", "2026-03-27", "2026-04-02", "2026-04-10"],
        )


if __name__ == "__main__":
    unittest.main()
