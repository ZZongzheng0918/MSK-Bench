from __future__ import annotations

import unittest


class PeakEfficiencyTest(unittest.TestCase):
    def test_peak_efficiency_uses_best_mean_return_step(self) -> None:
        from msk_bench.analysis.efficiency import peak_efficiency_steps

        rows = [
            {"step": 100, "mean_return": 10.0},
            {"step": 1000, "mean_return": 25.0},
            {"step": 10_000, "mean_return": 20.0},
        ]

        self.assertEqual(peak_efficiency_steps(rows), 1000)

    def test_peak_efficiency_supports_environment_step_alias(self) -> None:
        from msk_bench.analysis.efficiency import peak_efficiency_steps

        rows = [
            {"environment_step": 10, "mean_return": 0.2},
            {"environment_step": 100, "mean_return": 0.7},
        ]

        self.assertEqual(peak_efficiency_steps(rows), 100)

    def test_averages_seeds_and_breaks_ties_at_earliest_step(self):
        from msk_bench.analysis.efficiency import peak_efficiency_steps
        rows = [
            {"step": 200, "mean_return": 10},
            {"step": 100, "mean_return": 0},
            {"step": 100, "mean_return": 20},
            {"step": 300, "mean_return": float("nan")},
        ]
        self.assertEqual(peak_efficiency_steps(rows), 100)

    def test_missing_returns_are_not_reported_as_efficiency(self):
        from msk_bench.analysis.efficiency import peak_efficiency_steps
        with self.assertRaises(ValueError):
            peak_efficiency_steps([{"step": 100, "success_rate": 1}])


if __name__ == "__main__":
    unittest.main()
