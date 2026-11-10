"""Behavior checks for real datasets, calculations, plots, and practice code.

Run: python -m unittest discover -s tests -p 'test_ml_examples.py'
"""
import contextlib
import ast
import io
import itertools
import hashlib
import json
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
from sklearn.datasets import load_wine
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
    def test_beginner_programs_use_ordinary_loops_and_no_comprehensions(self):
        for unit in [3, 4, 5]:
            source = (ROOT / f"website/code/week{unit}.py").read_text()
            for node in ast.walk(ast.parse(source)):
                self.assertNotIsInstance(node, (ast.Lambda, ast.ListComp, ast.DictComp,
                                                ast.SetComp, ast.GeneratorExp),
                                         f"Unit {unit} should use ordinary steps and loops")

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

    def test_clustering_uses_real_features_and_the_displayed_ward_tree(self):
        result = run_question(3)
        wine = load_wine()
        self.assertEqual(result["X"].shape, (178, 13))
        np.testing.assert_allclose(result["dataset"].data, wine.data)
        self.assertEqual(result["columns"], wine.feature_names)
        for labels in result["labels_by_method"].values():
            self.assertEqual(len(labels), 178)
            self.assertTrue(np.isfinite(silhouette_score(result["X"], labels)))
        np.testing.assert_allclose(result["responsibilities"].sum(axis=1), 1)
        np.testing.assert_array_equal(result["labels_by_method"]["Ward"],
                                      fcluster(result["tree"], 3, criterion="maxclust") - 1)
        self.assertEqual(len(result["_figures"]), 4)

    def test_pca_feature_weights_and_variance_are_mathematically_consistent(self):
        result = run_question(3, 1)
        self.assertEqual(result["X"].shape, (178, 13))
        variance = result["pca"].explained_variance_ratio_
        np.testing.assert_allclose(variance.sum(), 1)
        np.testing.assert_allclose(result["cumulative"], np.cumsum(variance))
        self.assertGreaterEqual(result["cumulative"][result["k95"] - 1], 0.95)
        self.assertLess(result["cumulative"][result["k95"] - 2], 0.95)
        np.testing.assert_allclose(result["loadings"], result["pca"].components_[:2].T)
        reconstruction = result["pca"].inverse_transform(result["X_pca"])
        np.testing.assert_allclose(reconstruction, result["X"], atol=1e-12)
        self.assertEqual(len(result["_figures"]), 4)

    def test_sleep_data_provenance_and_recording_boundaries(self):
        path = ROOT / "website/data/sleep_edf.csv"
        metadata = json.loads(path.with_name("sleep_edf_provenance.json").read_text())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), metadata["csv_sha256"])
        data = pd.read_csv(path)
        self.assertEqual(data.shape, (21948, 19))
        self.assertFalse(data.isna().any().any())
        self.assertEqual((data.stage >= 0).sum(), 21947)
        self.assertEqual(set(data.stage), {-1, 0, 1, 2, 3, 4})
        self.assertEqual(len(metadata["source_files"]), 16)
        self.assertEqual(list(data.columns[4:]), metadata["features"])
        self.assertFalse(data.duplicated(["subject", "epoch"]).any())
        for subject, recording in data.groupby("subject"):
            np.testing.assert_array_equal(recording.epoch, np.arange(len(recording)))
            np.testing.assert_array_equal(recording.seconds, np.arange(len(recording)) * 30)
        self.assertEqual(sorted(data.subject.unique()), metadata["subjects"])

    def test_sleep_hmm_subject_isolation_outputs_and_independent_path_checks(self):
        result = run_question(4)
        self.assertEqual(len(result["_figures"]), 5)
        self.assertEqual(len(result["predictions"]), 21948)  # Includes the unscored epoch.
        self.assertTrue(result["predictions"]["decoded"].between(0, 4).all())
        self.assertNotIn("stage", result["features"])
        self.assertEqual(len(result["features"]), 15)
        for held_out, training in result["fold_training_subjects"].items():
            self.assertNotIn(held_out, training)
            self.assertEqual(len(training), 7)
        model = result["example_model"]
        for probabilities in [model["transition"], model["start"], model["prior"]]:
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertTrue((probabilities > 0).all())
            np.testing.assert_allclose(probabilities.sum(axis=-1), 1)
        training = result["df"]
        training = training[(training.subject != "SC400") & (training.stage >= 0)]
        np.testing.assert_allclose(model["centre"], training[result["features"]].mean())
        np.testing.assert_allclose(model["scale"], training[result["features"]].std(ddof=0))
        # Assert the actual, independently verified benchmark, not an inflated target.
        self.assertAlmostEqual(result["hmm_accuracy"], 0.8612566637809268, places=10)
        self.assertAlmostEqual(result["hmm_macro_f1"], 0.6960567151192041, places=10)
        np.testing.assert_allclose(result["posterior"].sum(axis=1), 1, atol=1e-8)
        # Check Gaussian density against a separate scipy implementation.
        from scipy.stats import norm
        values = result["example"][result["features"]].to_numpy()[:5]
        z = (values - model["centre"]) / model["scale"]
        expected = norm.logpdf(z[:, None, :], loc=model["means"], scale=np.sqrt(model["variances"])).sum(axis=2)
        np.testing.assert_allclose(result["log_emissions"](model, values), expected)
        # Enumerate every path in a small HMM to independently check both algorithms.
        rng = np.random.default_rng(42)
        for trial in range(4):
            start = rng.dirichlet(np.ones(3))
            transition = rng.dirichlet(np.ones(3), size=3)
            emissions = rng.normal(-2, 1, (5, 3))
            paths = list(itertools.product(range(3), repeat=5))
            weights = []
            for path in paths:
                weight = np.log(start[path[0]])
                for time in range(5):
                    weight += emissions[time, path[time]]
                    if time > 0:
                        weight += np.log(transition[path[time - 1], path[time]])
                weights.append(weight)
            weights = np.array(weights)
            np.testing.assert_array_equal(result["viterbi"](emissions, start, transition), paths[weights.argmax()])
            posterior, likelihood = result["forward_backward"](emissions, start, transition)
            normalizer = np.logaddexp.reduce(weights)
            self.assertAlmostEqual(likelihood, normalizer, places=12)
            for time in range(5):
                for state in range(3):
                    mask = np.array([path[time] == state for path in paths])
                    self.assertAlmostEqual(posterior[time, state], np.exp(weights[mask] - normalizer).sum(), places=12)
        emissions = np.tile([[-1., -100., -100.]], (3000, 1))
        start = np.ones(3) / 3
        transition = np.ones((3, 3)) / 3
        self.assertTrue((result["viterbi"](emissions, start, transition) == 0).all())
        posterior, likelihood = result["forward_backward"](emissions, start, transition)
        self.assertTrue(np.isfinite(likelihood))
        np.testing.assert_allclose(posterior.sum(axis=1), 1)
        self.assertEqual(len(result["viterbi"](np.empty((0, 3)), start, transition)), 0)
        self.assertEqual(result["forward_backward"](np.empty((0, 3)), start, transition)[1], 0)
        # Known training gaps and person boundaries cannot invent extra transitions.
        fake = training.head(7).copy()
        fake["subject"] = ["A", "A", "A", "A", "A", "B", "B"]
        fake["stage"] = [0, -1, 1, 2, 3, 4, 4]
        fitted = result["fit_hmm"](fake)
        expected_counts = np.ones((5, 5))
        expected_counts[1, 2] += 1
        expected_counts[2, 3] += 1
        expected_counts[4, 4] += 1
        np.testing.assert_allclose(fitted["transition"], expected_counts / expected_counts.sum(axis=1, keepdims=True))

    def test_sleep_explorer_matches_the_full_program_for_every_person(self):
        result = run_question(4)
        bridge = dict(result)
        exec(compile((ROOT / "website/code/sleep_bridge.py").read_text(), "sleep_bridge.py", "exec"), bridge)
        source = (ROOT / "website/code/week4.py").read_text()
        self.assertIn("# 2. Leave one person out", source)  # the page reuses everything above this line
        table = {row["Person"]: row for row in result["fold_scores"]}
        pooled = {key: np.zeros((5, 5), dtype=int) for key in ["majority", "independent", "decoded"]}
        for subject in result["subjects"]:
            fold = json.loads(bridge["explore"](subject))
            self.assertEqual(fold["subject"], subject)
            self.assertNotIn(subject, fold["training_subjects"])
            self.assertEqual(len(fold["training_subjects"]), 7)
            self.assertEqual(fold["training_segments"], int(((result["df"].subject != subject) & (result["df"].stage >= 0)).sum()))
            self.assertEqual(sum(fold["training_stage_counts"]), fold["training_segments"])
            for key in pooled:
                self.assertAlmostEqual(fold["models"][key]["accuracy"], table[subject][key + " accuracy"], places=12)
                self.assertAlmostEqual(fold["models"][key]["macro_f1"], table[subject][key + " macro F1"], places=12)
                pooled[key] += np.array(fold["models"][key]["confusion"])
            np.testing.assert_allclose(np.sum(fold["posterior"], axis=1), 1, atol=3e-3)  # values are rounded to 3 decimals
            np.testing.assert_allclose(np.sum(fold["transition"], axis=1), 1)
            self.assertEqual(len(fold["expert"]), fold["segments"])
            self.assertEqual(len(fold["decoded"]), fold["segments"])
        # Summing the eight held-out confusion matrices gives the benchmark of the full program.
        self.assertAlmostEqual(np.trace(pooled["decoded"]) / pooled["decoded"].sum(), 0.8612566637809268, places=10)

    def test_unit_five_uses_real_data_and_keeps_cv_within_training_set(self):
        cart = run_question(5)
        real = load_wine()
        self.assertEqual(cart["X"].shape, (178, 13))
        np.testing.assert_allclose(cart["X"], real.data)
        self.assertEqual(set(cart["y_test"]), {0, 1, 2})
        self.assertFalse(set(cart["X_train"].index) & set(cart["X_test"].index))
        for train_ids, validation_ids in cart["cv"].split(cart["X_train"], cart["y_train"]):
            fold_indices = cart["X_train"].iloc[validation_ids].index
            self.assertFalse(set(fold_indices) & set(cart["X_test"].index))
        self.assertGreaterEqual(len(cart["_figures"]), 3)
        ensemble = run_question(5, 1)
        np.testing.assert_array_equal(cart["X_test"].index, ensemble["X_test"].index)
        for model in ensemble["models"].values():
            predicted = model.predict(ensemble["X_test"])
            self.assertTrue(set(predicted) <= {0, 1, 2})
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
