"""Behavior checks for real datasets, calculations, plots, and practice code.

Run: python -m unittest discover -s tests -p 'test_ml_examples.py'
"""
import contextlib
import io
import itertools
import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster
from website.weather import use_style

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "website/data/seattle_weather.csv"


def run_question(unit, index=0):
    source = (ROOT / f"website/code/week{unit}.py").read_text()
    block = re.split(r"(?m)^# %% ", source)[index + 1]
    code = block.partition("\n")[2]
    namespace = {}
    figures = []
    def capture_plots():
        for number in plt.get_fignums():
            figure = plt.figure(number)
            figure.canvas.draw()
            figures.append(figure)
        plt.close("all")
    previous = Path.cwd()
    try:
        os.chdir(ROOT / "website")
        use_style()
        with contextlib.redirect_stdout(io.StringIO()), patch.object(plt, "show", side_effect=capture_plots):
            exec(compile(code, f"week{unit}.py", "exec"), namespace)
            namespace["_figures"] = figures
    finally:
        os.chdir(previous)
        plt.close("all")
    return namespace


class TeachingExamples(unittest.TestCase):
    def test_weather_is_the_real_daily_sequence(self):
        df = pd.read_csv(DATA)
        self.assertEqual(list(df.columns), ["date", "precipitation", "temp_max", "temp_min", "wind", "weather"])
        self.assertEqual(len(df), 1461)
        self.assertFalse(df.isna().any().any())
        dates = pd.to_datetime(df.date)
        self.assertEqual(df.date.iloc[0], "2012-01-01")
        self.assertEqual(df.date.iloc[-1], "2015-12-31")
        self.assertTrue(dates.diff().dropna().eq(pd.Timedelta(days=1)).all())
        self.assertTrue((df.temp_max >= df.temp_min).all())
        self.assertTrue((df.precipitation >= 0).all())
        self.assertTrue((df.wind > 0).all())
        self.assertEqual(set(df.weather), {"sun", "rain", "snow", "fog", "drizzle"})
        self.assertEqual(df.iloc[0].tolist(), ["2012-01-01", 0.0, 12.8, 5.0, 4.7, "drizzle"])

    def test_clustering_uses_real_features_and_correct_projected_covariances(self):
        result = run_question(3)
        wine = load_wine()
        self.assertEqual(result["X"].shape, (178, 13))
        np.testing.assert_allclose(result["dataset"].data, wine.data)
        self.assertEqual(result["columns"], wine.feature_names)
        for labels in result["labels_by_method"].values():
            self.assertEqual(len(labels), 178)
            self.assertTrue(np.isfinite(silhouette_score(result["X"], labels)))
        projected = result["projected_covariances"]
        projection = result["projection"].components_
        for component, covariance in enumerate(projected):
            full = result["component_covariance"](component)
            np.testing.assert_allclose(covariance, projection @ full @ projection.T)
        np.testing.assert_allclose(result["responsibilities"].sum(axis=1), 1)
        np.testing.assert_array_equal(result["labels_by_method"]["Ward"],
                                      fcluster(result["tree"], result["cut_height"], criterion="distance") - 1)
        self.assertGreaterEqual(len(result["_figures"]), 6)

    def test_pca_loadings_biplot_and_variance_are_mathematically_consistent(self):
        result = run_question(3, 1)
        self.assertEqual(result["X"].shape, (178, 13))
        variance = result["pca"].explained_variance_ratio_
        np.testing.assert_allclose(variance.sum(), 1)
        np.testing.assert_allclose(result["cumulative"], np.cumsum(variance))
        self.assertGreaterEqual(result["cumulative"][result["k95"] - 1], 0.95)
        self.assertLess(result["cumulative"][result["k95"] - 2], 0.95)
        correlation = np.corrcoef(result["X"].T, result["X_pca"].T)[:13, 13:16]
        np.testing.assert_allclose(result["correlation_loadings"][:, :3], correlation, atol=1e-12)
        reconstruction = result["X_pca"][:, :2] @ result["pca"].components_[:2]
        np.testing.assert_allclose(result["biplot_samples"] @ result["biplot_features"].T, reconstruction)
        self.assertEqual(len(result["_figures"]), 6)

    def test_hmm_probabilities_and_long_sequence_decoding(self):
        result = run_question(4)
        for frame in [result["transition"], result["emission"]]:
            self.assertTrue(np.isfinite(frame.values).all())
            self.assertTrue((frame.values > 0).all())
            np.testing.assert_allclose(frame.sum(axis=1), 1)
        # Repeated small ranges favor wet. Probability multiplication underflows
        # on this sequence and can silently switch the returned path to dry.
        self.assertEqual(set(result["viterbi"](["small"] * 2000)), {"wet"})
        self.assertEqual(result["viterbi"]([]), [])
        # Independently enumerate every path on a tiny sequence to verify that
        # decoding maximizes the complete path probability, including emissions.
        seen = ["large", "medium", "small", "small", "large"]
        paths = list(itertools.product(result["states"], repeat=len(seen)))
        def path_probability(path):
            score = result["start"][path[0]] * result["emission"].loc[path[0], seen[0]]
            for day in range(1, len(seen)):
                score *= result["transition"].loc[path[day - 1], path[day]]
                score *= result["emission"].loc[path[day], seen[day]]
            return score
        self.assertEqual(result["viterbi"](seen), list(max(paths, key=path_probability)))
        self.assertLess(result["train"].date.max(), result["test"].date.min())
        np.testing.assert_allclose(result["forecast_from_actual"].sum(axis=1), 1)
        self.assertEqual(len(result["forecast_predictions"]), len(result["test"]))
        self.assertTrue(np.isfinite(result["forecast_accuracy"]))

    def test_unit_five_uses_real_data_and_keeps_cv_within_training_set(self):
        cart = run_question(5)
        real = load_breast_cancer()
        self.assertEqual(cart["X"].shape, (569, 30))
        np.testing.assert_allclose(cart["X"], real.data)
        self.assertEqual(set(cart["y_test"]), set(real.target_names))
        self.assertFalse(set(cart["X_train"].index) & set(cart["X_test"].index))
        for train_ids, validation_ids in cart["cv"].split(cart["X_train"], cart["y_train"]):
            fold_indices = cart["X_train"].iloc[validation_ids].index
            self.assertFalse(set(fold_indices) & set(cart["X_test"].index))
        self.assertGreaterEqual(len(cart["_figures"]), 3)
        ensemble = run_question(5, 1)
        np.testing.assert_array_equal(cart["X_test"].index, ensemble["X_test"].index)
        for model in ensemble["models"].values():
            predicted = model.predict(ensemble["X_test"])
            self.assertTrue(set(predicted) <= set(real.target_names))
        self.assertEqual(len(ensemble["cv_results"]), len(ensemble["models"]))

    def test_forecasts_train_before_their_test_period(self):
        for index in [0, 1]:
            result = run_question(2, index)
            self.assertLess(result["y_train"].index.max(), result["y_test"].index.min())
            if index == 1:
                lower, upper = result["_figures"][-1].axes[0].get_ylim()
                self.assertLessEqual(lower, 0)
                self.assertGreaterEqual(upper, 1)


if __name__ == "__main__":
    unittest.main()
