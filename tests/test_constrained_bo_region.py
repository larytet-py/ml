import tempfile
import unittest
from pathlib import Path

from optimization.constrained_bo import ConstrainedBayesianOptimizer, ParamSpec, extract_region_summary
from optimization.goal_registry import build_goals


class ConstrainedBORegionTests(unittest.TestCase):
    def test_bo_runs_and_extracts_component_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trials_parquet = Path(tmp_dir) / "trials.parquet"
            eval_cache_parquet = Path(tmp_dir) / "eval_cache.parquet"

            # Feasible when x in [0.2, 0.8], with objective peak near 0.55.
            def evaluator(params):
                x = float(params["x"])
                avg_pnl = 100.0 - 400.0 * ((x - 0.55) ** 2)
                itm = 0.0 if 0.2 <= x <= 0.8 else 10.0
                drawdown = -50.0 if 0.2 <= x <= 0.8 else -500.0
                return {
                    "total": 30.0,
                    "wins": 20.0,
                    "win_rate": 2.0 / 3.0,
                    "itm_expiries": itm,
                    "itm_rate": itm / 30.0,
                    "total_pnl": avg_pnl * 30.0,
                    "avg_pnl": avg_pnl,
                    "median_pnl": avg_pnl,
                    "avg_return_on_spot": avg_pnl / 1000.0,
                    "max_drawdown": drawdown,
                }

            goals = build_goals(
                [
                    {"name": "itm_expiries", "kind": "constraint", "direction": "min", "target": 1.0},
                    {"name": "max_drawdown_abs", "kind": "constraint", "direction": "min", "target": 100.0},
                    {"name": "avg_pnl", "kind": "objective", "direction": "max", "target": None},
                ]
            )

            opt = ConstrainedBayesianOptimizer(
                search_space={"x": ParamSpec(low=0.0, high=1.0, is_int=False)},
                goals=goals,
                evaluator=evaluator,
                trials_parquet=str(trials_parquet),
                eval_cache_parquet=str(eval_cache_parquet),
                seed=7,
                n_random_init=8,
                candidate_pool_size=128,
                constraint_penalty=2000.0,
            )
            trials = opt.run(n_trials=24)

            self.assertTrue(trials_parquet.exists())
            self.assertTrue(eval_cache_parquet.exists())
            self.assertEqual(len(trials), 24)
            self.assertTrue(any(t.goals.feasible for t in trials))

            summary = extract_region_summary(
                trials=trials,
                search_space={"x": ParamSpec(low=0.0, high=1.0, is_int=False)},
                performance_tolerance_pct=5.0,
                min_component_fraction=0.10,
            )
            self.assertIn("selected_component", summary)
            self.assertIsNotNone(summary["selected_component"])
            self.assertGreaterEqual(summary["feasible_trial_count"], 1)


if __name__ == "__main__":
    unittest.main()
