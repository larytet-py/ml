import argparse
import tempfile
import unittest
from pathlib import Path

from optimize_option_strategy_region import _apply_window_bounds_from_yaml, _build_search_space


class OptimizeOptionStrategyRegionSearchSpaceTests(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            roc_window_min=1,
            roc_window_max=60,
            vol_window_min=5,
            vol_window_max=60,
            put_roc_threshold_min=-0.4,
            put_roc_threshold_max=0.4,
            downside_vol_threshold_min=0.0,
            downside_vol_threshold_max=1.0,
            put_accel_threshold_min=-0.2,
            put_accel_threshold_max=0.2,
            call_roc_threshold_min=-0.4,
            call_roc_threshold_max=0.4,
            upside_vol_threshold_min=0.0,
            upside_vol_threshold_max=1.0,
            call_accel_threshold_min=-0.2,
            call_accel_threshold_max=0.2,
        )

    def test_put_search_space_contains_only_used_dimensions(self):
        specs = _build_search_space(self._args(), side="put")
        self.assertEqual(set(specs.keys()), {"roc_window", "vol_window", "roc_threshold", "vol_threshold"})
        self.assertTrue(specs["roc_window"].is_int)
        self.assertEqual(specs["roc_window"].high, 60)

    def test_call_search_space_contains_only_used_dimensions(self):
        specs = _build_search_space(self._args(), side="call")
        self.assertEqual(set(specs.keys()), {"roc_window", "vol_window", "roc_threshold", "vol_threshold"})
        self.assertTrue(specs["vol_window"].is_int)


class OptimizeOptionStrategyRegionWindowYamlTests(unittest.TestCase):
    def test_window_bounds_are_loaded_from_yaml_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_path = Path(tmp_dir) / "windows.yaml"
            yaml_path.write_text(
                "\n".join(
                    [
                        "window_ranges:",
                        "  roc_window:",
                        "    min: 7",
                        "    max: 13",
                        "  vol_window:",
                        "    min: 9",
                        "    max: 21",
                    ]
                )
            )

            args = argparse.Namespace(
                window_config_yaml=str(yaml_path),
                roc_window_min=None,
                roc_window_max=None,
                accel_roc_window_min=None,
                accel_roc_window_max=None,
                accel_shift_window_min=None,
                accel_shift_window_max=None,
                vol_window_min=None,
                vol_window_max=None,
                corr_window_min=None,
                corr_window_max=None,
            )
            _apply_window_bounds_from_yaml(args)
            self.assertEqual(args.roc_window_min, 7)
            self.assertEqual(args.roc_window_max, 13)
            self.assertEqual(args.vol_window_min, 9)
            self.assertEqual(args.vol_window_max, 21)


if __name__ == "__main__":
    unittest.main()
