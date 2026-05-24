from __future__ import annotations

import unittest

import numpy as np

from gnss_denied_vio import SimulationConfig, run_simulation


class SimulationTest(unittest.TestCase):
    def test_visual_fusion_reduces_dropout_error(self) -> None:
        result = run_simulation(SimulationConfig(seed=7, duration_s=45.0, dropout_start_s=12.0, dropout_duration_s=20.0))
        self.assertLess(
            result.metrics["dropout_fused_position_rmse_m"],
            result.metrics["dropout_inertial_odom_position_rmse_m"],
        )
        self.assertGreater(result.metrics["visual_update_count"], 0)

    def test_gnss_is_suppressed_during_dropout(self) -> None:
        config = SimulationConfig(seed=3, duration_s=35.0, dropout_start_s=10.0, dropout_duration_s=15.0)
        result = run_simulation(config)
        dropout = result.measurements.dropout_mask
        self.assertFalse(np.any(result.measurements.gnss_available[dropout]))
        self.assertTrue(np.any(result.measurements.gnss_available[~dropout]))


if __name__ == "__main__":
    unittest.main()

