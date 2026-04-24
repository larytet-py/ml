import unittest

from backtest_weekly_option_reversal import run_backtest
from option_signal_notifier import SignalConfig, _evaluate_config, load_symbol_data_from_csv


CSV_PATH = "data/etfs.csv"
RISK_FREE_RATE = 0.04
MIN_PRICING_VOL = 0.10
CONTRACT_SIZE = 100
DOCUMENTED_SPY_PUT_ENTRY_DATES = {"2026-03-12", "2026-03-30"}


def _spy_put_config() -> SignalConfig:
    return SignalConfig(
        symbol="SPY",
        side="put",
        roc_lookback=3,
        vol_window=23,
        put_roc_threshold=-0.015766,
        call_roc_threshold=0.03,
        downside_vol_threshold=0.051323,
        upside_vol_threshold=0.20,
    )


def _evaluate_on(end_date: str):
    df = load_symbol_data_from_csv(CSV_PATH, "SPY", None, end_date)
    return _evaluate_config(
        cfg=_spy_put_config(),
        symbol_df=df,
        risk_free_rate=RISK_FREE_RATE,
        min_pricing_vol=MIN_PRICING_VOL,
        contract_size=CONTRACT_SIZE,
    )


class EvaluateConfigSpyPutTests(unittest.TestCase):
    # Dates based on the SPY put block comments in option_signal_notifier.config:
    # put 2026-03-12 2026-03-13
    # put 2026-03-30 2026-04-02
    def test_put_alerts_fire_on_documented_entry_dates(self):
        for alert_date in ("2026-03-12", "2026-03-30"):
            with self.subTest(alert_date=alert_date):
                self.assertIn(alert_date, DOCUMENTED_SPY_PUT_ENTRY_DATES)
                evaluation = _evaluate_on(alert_date)
                self.assertEqual(evaluation["date"], alert_date)
                self.assertTrue(evaluation["put_trigger"])
                self.assertEqual(len(evaluation["fired_signals"]), 1)
                self.assertEqual(evaluation["fired_signals"][0]["side"], "put")

    def test_put_alert_does_not_fire_on_2026_03_13_exit_day(self):
        evaluation = _evaluate_on("2026-03-13")
        self.assertFalse(evaluation["put_trigger"])
        self.assertEqual(evaluation["fired_signals"], [])

    def test_put_alert_does_not_refire_on_2026_04_02_exit_day(self):
        evaluation = _evaluate_on("2026-04-02")
        self.assertFalse(evaluation["put_trigger"])
        self.assertEqual(evaluation["fired_signals"], [])

    def test_notifier_put_trigger_matches_backtest_entries_for_sample_range(self):
        cfg = _spy_put_config()
        full_df = load_symbol_data_from_csv(CSV_PATH, "SPY", None, "2026-04-03")
        trades = run_backtest(
            df=full_df.copy(),
            side="put",
            roc_lookback=cfg.roc_lookback,
            put_roc_threshold=cfg.put_roc_threshold,
            call_roc_threshold=cfg.call_roc_threshold,
            vol_window=cfg.vol_window,
            downside_vol_threshold_annualized=cfg.downside_vol_threshold,
            upside_vol_threshold_annualized=cfg.upside_vol_threshold,
            risk_free_rate=RISK_FREE_RATE,
            min_pricing_vol_annualized=MIN_PRICING_VOL,
            contract_size=CONTRACT_SIZE,
            allow_overlap=False,
        )
        expected_entry_dates = {d.date().isoformat() for d in trades["entry_date"]}

        date_slice = full_df[(full_df["date"] >= "2026-03-10") & (full_df["date"] <= "2026-04-03")]
        for dt in date_slice["date"]:
            end_date = dt.date().isoformat()
            evaluation = _evaluate_on(end_date)
            with self.subTest(end_date=end_date):
                self.assertEqual(evaluation["date"], end_date)
                self.assertEqual(evaluation["put_trigger"], end_date in expected_entry_dates)


if __name__ == "__main__":
    unittest.main()
