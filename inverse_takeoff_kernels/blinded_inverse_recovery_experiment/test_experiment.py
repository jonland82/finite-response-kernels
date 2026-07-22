import unittest

import numpy as np

from run_experiment import (
    ALPHA,
    fixed_speed_check,
    generate_schema,
    recover_modes,
    tree_profile,
    verify_finite_horizon_family,
)


class ExperimentTests(unittest.TestCase):
    def test_finite_horizon_construction(self):
        result = verify_finite_horizon_family()
        self.assertTrue(result["prefix_equal"])
        self.assertTrue(result["later_separates"])

    def test_fixed_speed_examples(self):
        self.assertTrue(fixed_speed_check({1: 2}))
        self.assertTrue(fixed_speed_check({1: 1, 8: 128}))
        self.assertTrue(fixed_speed_check({8: 255, 9: 2}))

    def test_immediate_profile_tends_to_one(self):
        profile = tree_profile({1: 2}, 30)
        self.assertLess(abs(profile[-1] - 1.0), 1e-8)

    def test_matrix_pencil_recovers_two_modes(self):
        poles = np.asarray([0.83, -0.41])
        amplitudes = np.asarray([1.2, -0.3])
        indices = np.arange(80)[:, None]
        values = np.real((poles[None, :] ** indices) @ amplitudes)
        order, recovered, forecast = recover_modes(values[:64], 4, 16)
        self.assertEqual(order, 2)
        self.assertEqual(len(recovered), 2)
        self.assertLess(np.max(np.abs(np.sort(recovered) - np.sort(poles))), 1e-6)
        self.assertLess(np.max(np.abs(forecast - values[64:])), 1e-6)


if __name__ == "__main__":
    unittest.main()
