import unittest

import numpy as np
import pandas as pd

from data_pipeline import BANDS, _select_sql, prepare_photometry
from train import (
    CLASSES,
    FEATURE_SETS,
    bootstrap_classification,
    calibrated_probabilities,
    fit_isotonic_calibration,
)


def make_row():
    row = {"objid": 1, "ra": 123.0, "dec": 22.0, "class": "QSO", "redshift": 2.1}
    for index, band in enumerate(BANDS):
        row[f"psfMag_{band}"] = 20.0 - index
        row[f"psfMagErr_{band}"] = 0.03 + index * 0.01
        row[f"modelMag_{band}"] = 20.2 - index
        row[f"modelMagErr_{band}"] = 0.04 + index * 0.01
        row[f"extinction_{band}"] = 0.1 * (index + 1)
    return row


class PhotometryFeatureTests(unittest.TestCase):
    def test_sql_sample_order_is_deterministic_hash(self):
        query = _select_sql("QSO", 25)
        self.assertIn("ORDER BY CHECKSUM(s.specobjid), s.specobjid", query)
        self.assertNotIn("NEWID()", query)

    def test_dereddened_adjacent_psf_colour(self):
        prepared, _ = prepare_photometry(pd.DataFrame([make_row()]))
        expected_u_g = (20.0 - 0.1) - (19.0 - 0.2)
        self.assertAlmostEqual(prepared.iloc[0]["psf_color_u-g"], expected_u_g)

    def test_colour_error_propagation(self):
        prepared, _ = prepare_photometry(pd.DataFrame([make_row()]))
        expected = np.hypot(0.03, 0.04)
        self.assertAlmostEqual(prepared.iloc[0]["psf_color_err_u-g"], expected)

    def test_sentinel_magnitude_is_removed(self):
        row = make_row()
        row["psfMag_i"] = -9999
        prepared, report = prepare_photometry(pd.DataFrame([row]))
        self.assertEqual(len(prepared), 0)
        self.assertEqual(report["sentinel_magnitude_rows"], 1)

    def test_redshift_and_coordinates_are_not_features(self):
        for features in FEATURE_SETS.values():
            self.assertFalse(any("redshift" in feature.lower() for feature in features))
            self.assertFalse(any(feature in {"ra", "dec"} for feature in features))

    def test_isotonic_probabilities_are_normalized(self):
        labels = pd.Series(["STAR", "GALAXY", "QSO"] * 20)
        probabilities = np.tile(np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]), (20, 1))
        calibrators = fit_isotonic_calibration(probabilities, labels)
        calibrated = calibrated_probabilities(probabilities, calibrators)
        self.assertEqual(calibrated.shape, (len(labels), len(CLASSES)))
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)
        self.assertTrue(np.isfinite(calibrated).all())

    def test_bootstrap_intervals_cover_headline_class_metrics(self):
        labels = pd.Series(["STAR", "GALAXY", "QSO"] * 4)
        predictions = np.array(["STAR", "GALAXY", "STAR"] * 4)
        result = bootstrap_classification(labels, predictions, draws=50, seed=7)
        self.assertIn("macro_f1", result["intervals"])
        for class_name in CLASSES:
            for metric in ("precision", "recall", "f1"):
                interval = result["intervals"][f"{class_name}_{metric}"]
                self.assertLessEqual(interval["lower_95"], interval["upper_95"])


if __name__ == "__main__":
    unittest.main()