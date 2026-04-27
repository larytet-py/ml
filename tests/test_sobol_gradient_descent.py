import tempfile
import threading
import time
import unittest
from pathlib import Path

from optimization.constrained_bo import ParamSpec
from optimization.goal_registry import build_goals
from optimization.sobol_gradient_descent import ConstrainedSobolGradientOptimizer


class SobolGradientOptimizerTests(unittest.TestCase):
    def test_workers_enable_parallel_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trials_parquet = Path(tmp_dir) / "trials.parquet"
            eval_cache_parquet = Path(tmp_dir) / "eval_cache.parquet"

            active = 0
            max_active = 0
            lock = threading.Lock()

            def evaluator(params):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    x = float(params["x"])
                    # Keep each call alive briefly so parallel workers overlap.
                    time.sleep(0.01)
                    avg_pnl = 100.0 - 200.0 * ((x - 0.5) ** 2)
                    return {
                        "total": 20.0,
                        "wins": 10.0,
                        "win_rate": 0.5,
                        "itm_expiries": 0.0,
                        "itm_rate": 0.0,
                        "total_pnl": avg_pnl * 20.0,
                        "avg_pnl": avg_pnl,
                        "median_pnl": avg_pnl,
                        "avg_return_on_spot": avg_pnl / 1000.0,
                        "max_drawdown": -40.0,
                    }
                finally:
                    with lock:
                        active -= 1

            goals = build_goals(
                [
                    {"name": "itm_expiries", "kind": "constraint", "direction": "min", "target": 0.0},
                    {"name": "max_drawdown_abs", "kind": "objective", "direction": "min"},
                    {"name": "avg_pnl", "kind": "objective", "direction": "max"},
                ]
            )

            opt = ConstrainedSobolGradientOptimizer(
                search_space={"x": ParamSpec(low=0.0, high=1.0, is_int=False)},
                goals=goals,
                evaluator=evaluator,
                trials_parquet=str(trials_parquet),
                eval_cache_parquet=str(eval_cache_parquet),
                seed=11,
                sobol_samples=16,
                top_ratio=0.25,
                local_probe_per_seed=4,
                gradient_steps=1,
                workers=4,
                progress_interval_seconds=0.0,
            )
            payload = opt.run()

            selected_seed_count = len(payload["top_seed_trial_ids"])
            expected_total = 16 + selected_seed_count * (4 + (2 * 1 + 1))
            self.assertEqual(expected_total, len(payload["trials"]))
            selected_seed_ids = set(payload["top_seed_trial_ids"])
            for trial in payload["sobol_trials"]:
                if trial.trial_id not in selected_seed_ids:
                    continue
                self.assertGreater(float(trial.metrics["total"]), 3.0)
                self.assertGreater(float(trial.metrics["total_pnl"]), 0.0)
                self.assertEqual(float(trial.metrics["itm_expiries"]), 0.0)
            self.assertGreater(max_active, 1)
            self.assertTrue(trials_parquet.exists())
            self.assertTrue(eval_cache_parquet.exists())


if __name__ == "__main__":
    unittest.main()
