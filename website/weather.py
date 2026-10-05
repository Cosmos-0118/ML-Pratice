"""Loads the Seattle daily weather data and adds the few columns every notebook uses."""
import numpy as np
import pandas as pd

FEATURES = ["precipitation", "temp_max", "temp_min", "wind", "month_sin", "month_cos"]


def load(path="data/seattle_weather.csv"):
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
