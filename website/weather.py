"""Loads the Seattle daily weather data and adds the few columns every question uses."""
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["precipitation", "temp_max", "temp_min", "wind", "month_sin", "month_cos"]
DATA = Path(__file__).parent / "data" / "seattle_weather.csv"


def load(path=DATA):
    d = pd.read_csv(path, parse_dates=["date"])
    d["rain"] = (d.precipitation > 0).astype(int)      # 1 = it rained (or snowed) today
    angle = 2 * np.pi * d.date.dt.month / 12            # month as a circle, so December sits next to January
    d["month_sin"], d["month_cos"] = np.sin(angle), np.cos(angle)
    d["rain_tomorrow"] = d.rain.shift(-1)               # what we try to predict (the last day has none)
    d["temp_tomorrow"] = d.temp_max.shift(-1)
    return d


def split(d):
    """Train on 2012-2014, test on 2015: the model never sees the future."""
    d = d.dropna()
    return d[d.date < "2015-01-01"], d[d.date >= "2015-01-01"]


def use_style():
    """Clean, readable plots and tables (used by the notebooks and by the website)."""
    import matplotlib.pyplot as plt
    from cycler import cycler
    pd.set_option("display.width", 110)
    pd.set_option("display.max_columns", 20)
    plt.rcParams.update({
        "figure.figsize": (9, 3.6), "figure.constrained_layout.use": True,
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.alpha": .25, "grid.linewidth": .8,
        "legend.frameon": False, "lines.linewidth": 2,
        "axes.prop_cycle": cycler(color=["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#72b7b2", "#eeca3b", "#9d755d"]),
    })
