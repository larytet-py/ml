import unittest

from option_signal_notifier import SignalConfig, _evaluate_config, load_symbol_data_from_csv


CSV_PATH = "data/etfs.csv"
RISK_FREE_RATE = 0.04
MIN_PRICING_VOL = 0.10
CONTRACT_SIZE = 100


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
        for alert_date in ("2026-03-12", "2026-03-13", "2026-03-30"):
            with self.subTest(alert_date=alert_date):
                evaluation = _evaluate_on(alert_date)
                self.assertTrue(evaluation["put_trigger"])
                self.assertEqual(len(evaluation["fired_signals"]), 1)
                self.assertEqual(evaluation["fired_signals"][0]["side"], "put")

    def test_put_alert_does_not_refire_on_2026_04_02_exit_day(self):
        evaluation = _evaluate_on("2026-04-02")
        self.assertFalse(evaluation["put_trigger"])
        self.assertEqual(evaluation["fired_signals"], [])


if __name__ == "__main__":
    unittest.main()
