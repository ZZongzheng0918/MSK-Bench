from __future__ import annotations

import unittest


class EmgSimilarityEvaluatorTest(unittest.TestCase):
    def test_process_and_align_simulated_emg_splits_cycles_and_phase_aligns(self) -> None:
        import numpy as np

        from msk_bench.analysis.emg import process_and_align_simulated_emg

        reference = np.sin(np.linspace(0.0, 2.0 * np.pi, 101, endpoint=False))
        reference = reference - reference.min()
        simulated = np.tile(reference, 4)
        simulated = np.roll(simulated, 23)

        aligned, spread, score = process_and_align_simulated_emg(simulated, reference)

        self.assertEqual(aligned.shape, (101,))
        self.assertEqual(spread.shape, (101,))
        self.assertGreater(score, 0.99)
        self.assertLess(float(np.max(spread)), 1e-10)


if __name__ == "__main__":
    unittest.main()
