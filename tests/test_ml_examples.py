"""Behavior checks for the shared teaching data and the actual practice code.

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
from sklearn.metrics import silhouette_score, f1_score
from website.weather import use_style

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "website/data/weather_regimes.csv"


def run_question(unit, index=0):
    source = (ROOT / f"website/code/week{unit}.py").read_text()
    block = re.split(r"(?m)^# %% ", source)[index + 1]
    code = block.partition("\n")[2]
    namespace = {}
    previous = Path.cwd()
    try:
        os.chdir(ROOT / "website")
        use_style()
        with contextlib.redirect_stdout(io.StringIO()), patch.object(plt, "show"):
            exec(compile(code, f"week{unit}.py", "exec"), namespace)
            namespace["_figures"] = [plt.figure(number) for number in plt.get_fignums()]
    finally:
        os.chdir(previous)
        plt.close("all")
    return namespace


class TeachingExamples(unittest.TestCase):
    def test_dataset_has_daily_physical_measurements_and_all_regimes(self):
        df = pd.read_csv(DATA)
        self.assertEqual(list(df.columns), ["date", "precipitation", "temp_max", "temp_min", "wind", "weather"])
        self.assertEqual(len(df), 1461)
        self.assertFalse(df.isna().any().any())
        dates = pd.to_datetime(df.date)
        self.assertEqual(df.date.iloc[0], "2012-01-01")
        self.assertEqual(df.date.iloc[-1], "2015-12-31")
        self.assertTrue(dates.diff().dropna().eq(pd.Timedelta(days=1)).all())
        self.assertTrue((df.temp_max > df.temp_min).all())
        self.assertTrue((df.precipitation >= 0).all())
        self.assertTrue((df.wind > 0).all())
        for period in [df[df.date < "2015-01-01"], df[df.date >= "2015-01-01"]]:
            self.assertEqual(set(period.weather), {"sun", "cloudy", "rain", "snow"})
            self.assertGreater(period.weather.value_counts().min(), 20)

    def test_three_algorithms_find_separated_clusters_without_using_labels(self):
        result = run_question(3)
        self.assertEqual(result["columns"], ["precipitation", "temp_max", "temp_min", "wind"])
        for method in ["kmeans", "gmm", "hierarchical"]:
            labels = result["df"][method]
            self.assertEqual(labels.nunique(), 4)
            self.assertGreater(silhouette_score(result["X"], labels), 0.55)

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
        baseline = result["test"]["state"].value_counts(normalize=True).max()
        self.assertGreater(result["accuracy"], baseline + 0.1)

    def test_unit_five_classifies_every_regime(self):
        cart = run_question(5)
        self.assertEqual(set(cart["y_test"]), {"sun", "cloudy", "rain", "snow"})
        self.assertGreater(f1_score(cart["y_test"], cart["y_pred"], average="macro"), 0.85)
        ensemble = run_question(5, 1)
        left, right = ensemble["_figures"][-1].axes[0].get_xlim()
        self.assertLessEqual(left, 0)
        self.assertGreaterEqual(right, 1)
        for model in ensemble["models"].values():
            predicted = model.predict(ensemble["X_test"])
            self.assertGreater(f1_score(ensemble["y_test"], predicted, average="macro"), 0.85)

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
