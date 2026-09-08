from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import fit_session_boundary_recovery as recovery


def make_session(rewards, number, date):
    return pd.DataFrame(
        {
            "reward": rewards,
            "_session_number": number,
            "_session_index": number + 10,
            "_date": date,
            "_protocol_day": number,
            "_trial_in_session": np.arange(1, len(rewards) + 1),
        }
    )


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


class BoundaryDataTest(unittest.TestCase):
    def test_boundary_extraction_preserves_metadata_and_empirical_drops(self):
        previous = make_session([1, 1, 1, 1, 0] * 60, 1, "20230101")
        following = make_session(
            [1] * 30 + [0] * 20 + [1] * 250, 2, "20230102"
        )
        bins, metrics = recovery.extract_boundary_data([previous, following])
        row = metrics.iloc[0]
        self.assertAlmostEqual(row["PreBoundaryBaseline"], 0.8)
        self.assertAlmostEqual(row["InitialPostPerformance"], 0.6)
        self.assertEqual(row["AbsoluteDrop"], row["DropMagnitude"])
        self.assertAlmostEqual(row["PercentDrop"], 25.0)
        self.assertTrue(row["PercentDropValid"])
        self.assertEqual(row["PreviousSession"], 1)
        self.assertEqual(row["NextSession"], 2)
        self.assertEqual(list(bins["BinStart"]), list(range(0, 300, 20)))
        self.assertTrue((bins["NTrials"] == 20).all())

    def test_group_curve_is_boundary_weighted_not_animal_equal(self):
        rows = pd.DataFrame(
            {
                "Animal": [1, 1, 2],
                "Protocol": ["AB"] * 3,
                "Genotype": ["WT"] * 3,
                "BoundaryNumber": [1, 2, 1],
                "BinStart": [0] * 3,
                "BinEnd": [19] * 3,
                "BinCenter": [9.5] * 3,
                "Performance": [0.0, 0.0, 1.0],
            }
        )
        curve = recovery.aggregate_group_curve(rows)
        self.assertAlmostEqual(curve.iloc[0]["ObservedMean"], 1 / 3)
        self.assertEqual(curve.iloc[0]["NBoundaries"], 3)
        self.assertEqual(curve.iloc[0]["NAnimals"], 2)

    def test_animal_fit_pools_each_animals_boundaries_by_bin(self):
        starts = np.arange(0, 300, 20)
        ends = starts + 19
        performance = recovery.binned_model_prediction(
            starts, ends, np.array([0.8, 0.2, 75.0])
        )
        rows = []
        metrics = []
        for animal in (1, 2):
            for boundary in (1, 2):
                for start, end, value in zip(starts, ends, performance):
                    rows.append(
                        {
                            "Animal": animal,
                            "Protocol": "CD",
                            "Genotype": "HET",
                            "Gender": "F",
                            "BoundaryNumber": boundary,
                            "BinStart": start,
                            "BinEnd": end,
                            "BinCenter": (start + end) / 2,
                            "Performance": value,
                        }
                    )
                metrics.append(
                    {
                        "Animal": animal,
                        "Protocol": "CD",
                        "BoundaryNumber": boundary,
                        "PreBoundaryBaseline": 0.8,
                        "DropMagnitude": 0.2,
                        "AbsoluteDrop": 0.2,
                        "PercentDrop": 25.0,
                        "PercentDropValid": True,
                    }
                )
        fits = recovery.fit_animal_curves(
            pd.DataFrame(rows), pd.DataFrame(metrics)
        )
        self.assertEqual(len(fits), 2)
        self.assertTrue(fits["FitSuccess"].all())
        self.assertTrue((fits["NBoundaries"] == 2).all())
        self.assertTrue((fits["NFitBins"] == 15).all())
        np.testing.assert_allclose(fits["A"], 0.2, atol=1e-5)
        np.testing.assert_allclose(fits["MeanPercentDrop"], 25.0)


if __name__ == "__main__":
    unittest.main()
