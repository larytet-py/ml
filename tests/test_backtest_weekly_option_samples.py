import unittest

from backtest_weekly_option_acceleration_reversal import run_backtest as run_accel_backtest
from backtest_weekly_option_acceleration_reversal import summarize_trades as summarize_accel_trades
from backtest_weekly_option_reversal import load_symbol_data
from backtest_weekly_option_reversal import run_backtest as run_roc_backtest
from backtest_weekly_option_reversal import summarize_trades as summarize_roc_trades
from backtest_weekly_option_roc_accel_reversal import run_backtest as run_roc_accel_backtest
from backtest_weekly_option_roc_accel_reversal import summarize_trades as summarize_roc_accel_trades


CSV_PATH = "tests/fixtures/etfs_golden_small.csv"
RISK_FREE_RATE = 0.04
MIN_PRICING_VOL = 0.10
CONTRACT_SIZE = 100


class BacktestTableSamplesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spy_df = load_symbol_data(CSV_PATH, "SPY", None, None)
        cls.qqq_df = load_symbol_data(CSV_PATH, "QQQ", None, None)
        if cls.spy_df.empty or cls.qqq_df.empty:
            raise AssertionError("Golden fixture must include at least two symbols with data (SPY and QQQ).")
        if len(cls.spy_df) < 12:
            raise AssertionError("Golden fixture must include enough SPY rows for at least two rolling windows.")

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
        # Golden sample with small windows for fast unit tests.
        trades = run_roc_backtest(
            df=self.spy_df.copy(),
            side="call",
            roc_lookback=2,
            put_roc_threshold=-0.03,
            call_roc_threshold=0.0,
            vol_window=2,
            downside_vol_threshold_annualized=0.2,
            upside_vol_threshold_annualized=0.03,
            risk_free_rate=RISK_FREE_RATE,
            min_pricing_vol_annualized=MIN_PRICING_VOL,
            contract_size=CONTRACT_SIZE,
            allow_overlap=False,
        )
        metrics = summarize_roc_trades(trades)

        self._assert_common_metrics(
            metrics=metrics,
            trades=trades,
            expected_trades=5,
            expected_win_rate_pct=60.00,
            expected_itm_count=3,
            expected_itm_rate_pct=60.00,
            expected_total_pnl=-1166.31,
            expected_avg_pnl=-233.26,
            expected_median_pnl=295.08,
            expected_avg_return_pct=-0.3501,
            expected_max_drawdown=-2152.44,
        )

        expected_recent_entry_dates = [
            "2026-02-09",
            "2026-03-17",
            "2026-04-01",
            "2026-04-06",
            "2026-04-15",
        ]
        expected_recent_exit_dates = [
            "2026-02-13",
            "2026-03-20",
            "2026-04-02",
            "2026-04-10",
            "2026-04-17",
        ]
        self.assertEqual(
            [d.date().isoformat() for d in trades["entry_date"].tail(5)],
            expected_recent_entry_dates,
        )
        self.assertEqual(
            [d.date().isoformat() for d in trades["exit_date"].tail(5)],
            expected_recent_exit_dates,
        )

    def test_acceleration_sample_from_config_matches_table(self):
        # Golden sample with small windows for fast unit tests.
        trades = run_accel_backtest(
            df=self.spy_df.copy(),
            side="call",
            accel_window=2,
            put_accel_threshold=-0.03,
            call_accel_threshold=0.0,
            vol_window=3,
            downside_vol_threshold_annualized=0.2,
            upside_vol_threshold_annualized=0.03,
            risk_free_rate=RISK_FREE_RATE,
            min_pricing_vol_annualized=MIN_PRICING_VOL,
            contract_size=CONTRACT_SIZE,
            allow_overlap=False,
        )
        metrics = summarize_accel_trades(trades)

        self._assert_common_metrics(
            metrics=metrics,
            trades=trades,
            expected_trades=5,
            expected_win_rate_pct=60.00,
            expected_itm_count=2,
            expected_itm_rate_pct=40.00,
            expected_total_pnl=-1256.10,
            expected_avg_pnl=-251.22,
            expected_median_pnl=251.15,
            expected_avg_return_pct=-0.3757,
            expected_max_drawdown=-2099.94,
        )

        self.assertEqual(
            [d.date().isoformat() for d in trades["entry_date"]],
            [
                "2026-02-10",
                "2026-03-18",
                "2026-03-25",
                "2026-04-06",
                "2026-04-16",
            ],
        )
        self.assertEqual(
            [d.date().isoformat() for d in trades["exit_date"]],
            [
                "2026-02-13",
                "2026-03-20",
                "2026-03-27",
                "2026-04-10",
                "2026-04-17",
            ],
        )

    def test_roc_acceleration_sample_from_config_matches_table(self):
        # Golden sample with small windows for fast unit tests.
        trades = run_roc_accel_backtest(
            df=self.spy_df.copy(),
            side="put",
            roc_lookback=3,
            accel_window=2,
            put_roc_threshold=-0.01,
            call_roc_threshold=0.03,
            put_accel_threshold=0.004,
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
            expected_trades=5,
            expected_win_rate_pct=40.00,
            expected_itm_count=3,
            expected_itm_rate_pct=60.00,
            expected_total_pnl=-3891.58,
            expected_avg_pnl=-778.32,
            expected_median_pnl=-1243.86,
            expected_avg_return_pct=-1.1654,
            expected_max_drawdown=-4582.91,
        )

        self.assertEqual(
            [d.date().isoformat() for d in trades["entry_date"]],
            ["2026-02-17", "2026-03-09", "2026-03-16", "2026-03-24", "2026-03-30"],
        )
        self.assertEqual(
            [d.date().isoformat() for d in trades["exit_date"]],
            ["2026-02-20", "2026-03-13", "2026-03-20", "2026-03-27", "2026-04-02"],
        )


if __name__ == "__main__":
    unittest.main()
