import argparse
import tempfile
import unittest
from pathlib import Path

from backtest_weekly_option_acceleration_reversal import run_backtest as run_accel_backtest
from backtest_weekly_option_reversal import run_backtest as run_roc_backtest
from option_signal_notifier import SignalConfig, _evaluate_config, _load_configs, load_symbol_data_from_csv


CSV_PATH = "data/etfs.csv"
RISK_FREE_RATE = 0.04
MIN_PRICING_VOL = 0.10
CONTRACT_SIZE = 100
EXPECTED_ACCEL_RECENT_PUT_ENTRY_DATES = {
    "2026-04-07",
    "2026-04-14",
}


def _spy_put_roc_config() -> SignalConfig:
    return SignalConfig(
        symbol="SPY",
        side="put",
        signal_model="roc",
        roc_lookback=3,
        accel_window=5,
        vol_window=23,
        put_roc_threshold=-0.015766,
        call_roc_threshold=0.03,
        put_accel_threshold=-0.03,
        call_accel_threshold=0.03,
        downside_vol_threshold=0.051323,
        upside_vol_threshold=0.20,
    )


def _spy_put_accel_config() -> SignalConfig:
    return SignalConfig(
        symbol="SPY",
        side="put",
        signal_model="accel",
        roc_lookback=3,
        accel_window=4,
        vol_window=23,
        put_roc_threshold=-0.03,
        call_roc_threshold=0.03,
        put_accel_threshold=-0.009787,
        call_accel_threshold=0.03,
        downside_vol_threshold=0.086187,
        upside_vol_threshold=0.20,
    )


def _evaluate_on(cfg: SignalConfig, end_date: str):
    df = load_symbol_data_from_csv(CSV_PATH, "SPY", None, end_date)
    return _evaluate_config(
        cfg=cfg,
        symbol_df=df,
        risk_free_rate=RISK_FREE_RATE,
        min_pricing_vol=MIN_PRICING_VOL,
        contract_size=CONTRACT_SIZE,
    )


class EvaluateConfigRocParityTests(unittest.TestCase):
    def test_notifier_put_roc_trigger_matches_roc_backtest_entries(self):
        cfg = _spy_put_roc_config()
        full_df = load_symbol_data_from_csv(CSV_PATH, "SPY", None, "2026-04-03")
        trades = run_roc_backtest(
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
            evaluation = _evaluate_on(cfg, end_date)
            with self.subTest(end_date=end_date):
                self.assertEqual(evaluation["date"], end_date)
                self.assertEqual(evaluation["signal_model"], "roc")
                self.assertEqual(evaluation["put_trigger"], end_date in expected_entry_dates)

    def test_notifier_does_not_fire_on_friday_close(self):
        cfg = _spy_put_roc_config()

        evaluation = _evaluate_on(cfg, "2026-03-20")

        self.assertFalse(evaluation["put_trigger"])
        self.assertEqual(len(evaluation["fired_signals"]), 0)

    def test_backtest_skips_friday_close_entry(self):
        cfg = _spy_put_roc_config()
        full_df = load_symbol_data_from_csv(CSV_PATH, "SPY", None, "2026-03-27")

        trades = run_roc_backtest(
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
            allow_overlap=True,
        )

        target_date = "2026-03-20"
        self.assertNotIn(target_date, {d.date().isoformat() for d in trades["entry_date"]})


class EvaluateConfigAccelParityTests(unittest.TestCase):
    def test_notifier_put_accel_trigger_matches_accel_backtest_entries(self):
        cfg = _spy_put_accel_config()
        full_df = load_symbol_data_from_csv(CSV_PATH, "SPY", None, "2026-04-17")
        trades = run_accel_backtest(
            df=full_df.copy(),
            side="put",
            accel_window=cfg.accel_window,
            put_accel_threshold=cfg.put_accel_threshold,
            call_accel_threshold=cfg.call_accel_threshold,
            vol_window=cfg.vol_window,
            downside_vol_threshold_annualized=cfg.downside_vol_threshold,
            upside_vol_threshold_annualized=cfg.upside_vol_threshold,
            risk_free_rate=RISK_FREE_RATE,
            min_pricing_vol_annualized=MIN_PRICING_VOL,
            contract_size=CONTRACT_SIZE,
            allow_overlap=False,
        )
        expected_entry_dates = {d.date().isoformat() for d in trades["entry_date"]}
        expected_recent_entry_dates = {d for d in expected_entry_dates if d >= "2026-04-01"}
        self.assertEqual(expected_recent_entry_dates, EXPECTED_ACCEL_RECENT_PUT_ENTRY_DATES)

        for end_date in sorted(EXPECTED_ACCEL_RECENT_PUT_ENTRY_DATES):
            evaluation = _evaluate_on(cfg, end_date)
            with self.subTest(end_date=end_date):
                self.assertEqual(evaluation["date"], end_date)
                self.assertEqual(evaluation["signal_model"], "accel")
                self.assertTrue(evaluation["put_trigger"])


class ConfigParsingTests(unittest.TestCase):
    def test_config_accepts_accel_threshold_flags(self):
        cfg_text = (
            "--symbol SPY --side put --signal-model accel --accel-window 9 --vol-window 21 "
            "--put-accel-threshold -0.012 --downside-vol-threshold 0.07\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "notifier.config"
            cfg_path.write_text(cfg_text)
            args = argparse.Namespace(
                config=str(cfg_path),
                symbol="SPY",
                side="both",
                signal_model="roc",
                roc_lookback=5,
                accel_window=5,
                vol_window=20,
                put_roc_threshold=-0.03,
                call_roc_threshold=0.03,
                put_accel_threshold=-0.03,
                call_accel_threshold=0.03,
                downside_vol_threshold=0.20,
                upside_vol_threshold=0.20,
                allow_friday_close_entry=False,
            )
            configs = _load_configs(args)

        self.assertEqual(len(configs), 1)
        cfg = configs[0]
        self.assertEqual(cfg.signal_model, "accel")
        self.assertEqual(cfg.accel_window, 9)
        self.assertAlmostEqual(cfg.put_accel_threshold, -0.012)
        self.assertAlmostEqual(cfg.downside_vol_threshold, 0.07)


if __name__ == "__main__":
    unittest.main()
