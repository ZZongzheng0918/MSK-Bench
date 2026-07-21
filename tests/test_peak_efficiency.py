from __future__ import annotations

import unittest


class PeakEfficiencyTest(unittest.TestCase):
    def test_peak_efficiency_uses_max_logged_step(self) -> None:
        from msk_bench.analysis.efficiency import peak_efficiency_steps

        rows = [
            {"step": 100, "mean_return": 10.0},
            {"step": 1000, "mean_return": 25.0},
            {"step": 10_000, "mean_return": 20.0},
        ]

        self.assertEqual(peak_efficiency_steps(rows), 10_000)

    def test_peak_efficiency_supports_environment_step_alias(self) -> None:
        from msk_bench.analysis.efficiency import peak_efficiency_steps

        rows = [
            {"environment_step": 10, "success_rate": 0.2},
            {"environment_step": 100, "success_rate": 0.7},
        ]

        self.assertEqual(peak_efficiency_steps(rows), 100)


if __name__ == "__main__":
    unittest.main()
