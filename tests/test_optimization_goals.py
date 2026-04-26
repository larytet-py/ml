import unittest

from optimization.goal_registry import build_goals, evaluate_goals


class OptimizationGoalTests(unittest.TestCase):
    def test_goal_registry_and_evaluation(self):
        goals = build_goals(
            [
                {"name": "itm_expiries", "kind": "constraint", "direction": "min", "target": 3},
                {"name": "max_drawdown_abs", "kind": "constraint", "direction": "min", "target": 100},
                {"name": "avg_pnl", "kind": "objective", "direction": "max", "target": None},
            ]
        )
        metrics = {"itm_expiries": 2.0, "max_drawdown": -80.0, "avg_pnl": 12.5}
        ev = evaluate_goals(goals, metrics)
        self.assertTrue(ev.feasible)
        self.assertAlmostEqual(ev.objective_score, 12.5)

        bad = {"itm_expiries": 5.0, "max_drawdown": -120.0, "avg_pnl": 50.0}
        ev_bad = evaluate_goals(goals, bad)
        self.assertFalse(ev_bad.feasible)
        self.assertGreater(ev_bad.violations["itm_expiries"], 0)
        self.assertGreater(ev_bad.violations["max_drawdown_abs"], 0)


if __name__ == "__main__":
    unittest.main()
