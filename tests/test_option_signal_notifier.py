import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from backtest_weekly_option_acceleration_reversal import run_backtest as run_accel_backtest
from backtest_weekly_option_roc_accel_reversal import run_backtest as run_roc_accel_backtest
from backtest_weekly_option_reversal import run_backtest as run_roc_backtest
from option_signal_notifier import (
    SignalConfig,
    _choose_atm_from_chain,
    _evaluate_config,
    _fetch_marketdata_option_candle,
    _fetch_marketdata_option_quote_snapshot,
    _load_configs,
    load_symbol_data_from_csv,
)


CSV_PATH = "data/etfs.csv"
RISK_FREE_RATE = 0.04
MIN_PRICING_VOL = 0.10
CONTRACT_SIZE = 100

def _spy_put_roc_config() -> SignalConfig:
    return SignalConfig(
        symbol="SPY",
        side="put",
        signal_model="roc",
        roc_lookback=3,
        accel_window=5,
        vol_window=23,
        roc_comparator="below",
        roc_threshold=-0.015766,
        accel_comparator=None,
        accel_threshold=None,
        vol_comparator="above",
        vol_threshold=0.051323,
    )


def _spy_call_accel_config() -> SignalConfig:
    return SignalConfig(
        symbol="SPY",
        side="call",
        signal_model="accel",
        roc_lookback=3,
        accel_window=53,
        vol_window=14,
        roc_comparator=None,
        roc_threshold=None,
        accel_comparator="below",
        accel_threshold=-0.013715,
        vol_comparator="above",
        vol_threshold=0.046678,
    )


def _spy_put_roc_accel_config() -> SignalConfig:
    return SignalConfig(
        symbol="SPY",
        side="put",
        signal_model="accel-roc",
        roc_lookback=3,
        accel_window=4,
        vol_window=23,
        roc_comparator="below",
        roc_threshold=-0.015766,
        accel_comparator="above",
        accel_threshold=-0.009787,
        vol_comparator=None,
        vol_threshold=None,
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
            put_roc_threshold=cfg.roc_threshold,
            call_roc_threshold=0.03,
            vol_window=cfg.vol_window,
            downside_vol_threshold_annualized=cfg.vol_threshold,
            upside_vol_threshold_annualized=0.20,
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
            put_roc_threshold=cfg.roc_threshold,
            call_roc_threshold=0.03,
            vol_window=cfg.vol_window,
            downside_vol_threshold_annualized=cfg.vol_threshold,
            upside_vol_threshold_annualized=0.20,
            risk_free_rate=RISK_FREE_RATE,
            min_pricing_vol_annualized=MIN_PRICING_VOL,
            contract_size=CONTRACT_SIZE,
            allow_overlap=True,
        )

        target_date = "2026-03-20"
        self.assertNotIn(target_date, {d.date().isoformat() for d in trades["entry_date"]})


class EvaluateConfigAccelParityTests(unittest.TestCase):
    def test_notifier_call_accel_trigger_matches_accel_backtest_entries(self):
        cfg = _spy_call_accel_config()
        full_df = load_symbol_data_from_csv(CSV_PATH, "SPY", None, "2026-04-17")
        trades = run_accel_backtest(
            df=full_df.copy(),
            side="call",
            accel_window=cfg.accel_window,
            put_accel_threshold=-0.009787,
            call_accel_threshold=cfg.accel_threshold,
            vol_window=cfg.vol_window,
            downside_vol_threshold_annualized=0.086187,
            upside_vol_threshold_annualized=cfg.vol_threshold,
            risk_free_rate=RISK_FREE_RATE,
            min_pricing_vol_annualized=MIN_PRICING_VOL,
            contract_size=CONTRACT_SIZE,
            allow_overlap=False,
        )
        expected_entry_dates = {d.date().isoformat() for d in trades["entry_date"]}

        date_slice = full_df[(full_df["date"] >= "2026-03-01") & (full_df["date"] <= "2026-04-17")]
        for dt in date_slice["date"]:
            end_date = dt.date().isoformat()
            evaluation = _evaluate_on(cfg, end_date)
            with self.subTest(end_date=end_date):
                self.assertEqual(evaluation["date"], end_date)
                self.assertEqual(evaluation["signal_model"], "accel")
                self.assertEqual(evaluation["call_trigger"], end_date in expected_entry_dates)


class EvaluateConfigRocAccelParityTests(unittest.TestCase):
    def test_notifier_put_roc_accel_trigger_matches_roc_accel_backtest_entries(self):
        cfg = _spy_put_roc_accel_config()
        full_df = load_symbol_data_from_csv(CSV_PATH, "SPY", None, "2026-04-17")
        trades = run_roc_accel_backtest(
            df=full_df.copy(),
            side="put",
            roc_lookback=cfg.roc_lookback,
            accel_window=cfg.accel_window,
            put_roc_threshold=cfg.roc_threshold,
            call_roc_threshold=0.03,
            put_accel_threshold=-0.009787,
            call_accel_threshold=0.03,
            risk_free_rate=RISK_FREE_RATE,
            min_pricing_vol_annualized=MIN_PRICING_VOL,
            contract_size=CONTRACT_SIZE,
            allow_overlap=False,
        )
        expected_entry_dates = {d.date().isoformat() for d in trades["entry_date"]}

        date_slice = full_df[(full_df["date"] >= "2026-03-10") & (full_df["date"] <= "2026-04-17")]
        for dt in date_slice["date"]:
            end_date = dt.date().isoformat()
            evaluation = _evaluate_on(cfg, end_date)
            with self.subTest(end_date=end_date):
                self.assertEqual(evaluation["date"], end_date)
                self.assertEqual(evaluation["signal_model"], "accel-roc")
                self.assertEqual(evaluation["put_trigger"], end_date in expected_entry_dates)


class ConfigParsingTests(unittest.TestCase):
    def test_config_accepts_accel_threshold_flags(self):
        cfg_text = (
            "--symbol SPY --side put --signal-model accel --accel-window 9 --vol-window-size 21 "
            "--accel-comparator above --accel-threshold -0.012 --vol-comparator above --vol-threshold 0.07\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "notifier.config"
            cfg_path.write_text(cfg_text)
            args = argparse.Namespace(
                config=str(cfg_path),
                symbol="SPY",
                side="both",
                signal_model="roc",
                roc_window_size=5,
                accel_window=5,
                vol_window_size=20,
                roc_comparator=None,
                roc_threshold=None,
                accel_comparator=None,
                accel_threshold=None,
                vol_comparator=None,
                vol_threshold=None,
                allow_friday_close_entry=False,
            )
            configs = _load_configs(args)

        self.assertEqual(len(configs), 1)
        cfg = configs[0]
        self.assertEqual(cfg.signal_model, "accel")
        self.assertEqual(cfg.accel_window, 9)
        self.assertEqual(cfg.accel_comparator, "above")
        self.assertAlmostEqual(cfg.accel_threshold, -0.012)
        self.assertEqual(cfg.vol_comparator, "above")
        self.assertAlmostEqual(cfg.vol_threshold, 0.07)

    def test_config_accepts_accel_roc_alias_and_normalizes(self):
        cfg_text = (
            "--symbol SPY --side put --signal-model accel/roc --roc-window-size 7 --accel-window 11 "
            "--roc-comparator below --roc-threshold -0.02 --accel-comparator below --accel-threshold -0.01\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "notifier.config"
            cfg_path.write_text(cfg_text)
            args = argparse.Namespace(
                config=str(cfg_path),
                symbol="SPY",
                side="both",
                signal_model="roc",
                roc_window_size=5,
                accel_window=5,
                vol_window_size=20,
                roc_comparator=None,
                roc_threshold=None,
                accel_comparator=None,
                accel_threshold=None,
                vol_comparator=None,
                vol_threshold=None,
                allow_friday_close_entry=False,
            )
            configs = _load_configs(args)

        self.assertEqual(len(configs), 1)
        cfg = configs[0]
        self.assertEqual(cfg.signal_model, "accel-roc")
        self.assertEqual(cfg.roc_lookback, 7)
        self.assertEqual(cfg.accel_window, 11)


class MarketData404HandlingTests(unittest.TestCase):
    def test_option_candle_404_returns_none(self):
        response = Mock()
        response.status_code = 404
        with patch("option_signal_notifier._marketdata_get", side_effect=requests.HTTPError(response=response)):
            candle = _fetch_marketdata_option_candle("GDX260417C00098000", "2026-04-15")
        self.assertIsNone(candle)

    def test_option_quote_404_returns_empty_dict(self):
        response = Mock()
        response.status_code = 404
        with patch("option_signal_notifier._marketdata_get", side_effect=requests.HTTPError(response=response)):
            quote = _fetch_marketdata_option_quote_snapshot("GDX260417C00098000", "2026-04-15")
        self.assertEqual(quote, {})


class MarketDataChainSelectionTests(unittest.TestCase):
    def test_choose_atm_skips_same_day_expiration_when_trade_date_provided(self):
        chain_payload = {
            "optionSymbol": ["IWM260428C00277000", "IWM260429C00277000"],
            "side": ["call", "call"],
            "strike": [277, 277],
            "expiration": [1777334400, 1777420800],  # 2026-04-28, 2026-04-29
        }
        chosen = _choose_atm_from_chain(chain_payload, side="call", spot=277.0, trade_date="2026-04-28")
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["optionSymbol"], "IWM260429C00277000")


if __name__ == "__main__":
    unittest.main()
