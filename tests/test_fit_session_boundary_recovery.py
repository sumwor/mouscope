from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import fit_session_boundary_recovery as recovery


class ExponentialRecoveryFitTest(unittest.TestCase):
    def test_binned_prediction_is_mean_over_integer_trials(self):
        got = recovery.binned_model_prediction(
            np.array([0]),
            np.array([19]),
            np.array([0.8, 0.2, 75.0]),
        )
        expected = np.mean(0.8 - 0.2 * np.exp(-np.arange(20) / 75.0))
        np.testing.assert_allclose(got, [expected])

    def test_seeded_synthetic_fit_recovers_parameters(self):
        rng = np.random.default_rng(240513)
        starts = np.arange(0, 300, 20)
        ends = starts + 19
        expected = recovery.binned_model_prediction(
            starts, ends, np.array([0.8, 0.2, 75.0])
        )
        curve = pd.DataFrame(
            {
                "BinStart": starts,
                "BinEnd": ends,
                "ObservedMean": expected + rng.normal(0, 0.004, len(starts)),
            }
        )
        fit = recovery.fit_exponential_recovery(curve)
        self.assertTrue(fit["FitSuccess"])
        self.assertLess(abs(fit["P_inf"] - 0.8), 0.04)
        self.assertLess(abs(fit["A"] - 0.2), 0.05)
        self.assertLess(abs(fit["lambda"] - 75), 30)
        self.assertEqual(fit["NFitBins"], 15)
        self.assertGreater(fit["R2"], 0.9)

    def test_fit_requires_eight_finite_bins(self):
        curve = pd.DataFrame(
            {
                "BinStart": np.arange(0, 140, 20),
                "BinEnd": np.arange(19, 159, 20),
                "ObservedMean": np.linspace(0.5, 0.7, 7),
            }
        )
        fit = recovery.fit_exponential_recovery(curve)
        self.assertFalse(fit["FitSuccess"])
        self.assertEqual(fit["NFitBins"], 7)
        self.assertTrue(np.isnan(fit["P_inf"]))
        self.assertTrue(fit["BoundaryWarning"])

    def test_warning_flags_report_each_parameter_reason(self):
        warnings = recovery.parameter_warnings(
            np.array([0.999, 0.001, 299.5]), True
        )
        self.assertEqual(
            warnings,
            {
                "PInfWarning": True,
                "AWarning": True,
                "LambdaWarning": True,
                "P0Warning": False,
                "BoundaryWarning": True,
            },
        )

    def test_warning_flags_report_out_of_range_fitted_p0(self):
        warnings = recovery.parameter_warnings(
            np.array([0.2, 0.5, 75.0]), True
        )
        self.assertFalse(warnings["PInfWarning"])
        self.assertFalse(warnings["AWarning"])
        self.assertFalse(warnings["LambdaWarning"])
        self.assertTrue(warnings["P0Warning"])
        self.assertTrue(warnings["BoundaryWarning"])


if __name__ == "__main__":
    unittest.main()
