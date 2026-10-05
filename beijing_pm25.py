"""Shared loader for the Beijing PM2.5 data set used in every weekly notebook.

Source : UCI Machine Learning Repository, "Beijing PM2.5 Data" (id 381), licence CC BY 4.0.
         Liang, X. et al. (2015) "Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating",
         Proc. R. Soc. A 471: 20150257.  PM2.5 = US Embassy in Beijing; weather = Beijing Capital International Airport.
Mirror : https://github.com/jbrownlee/Datasets (pollution.csv, byte-for-byte the UCI file) – downloaded once into ./data/.

Columns (hourly, 2010-01-01 00:00 → 2014-12-31 23:00, 43 824 rows)
    pm25   PM2.5 concentration, µg/m³ (2 067 hours missing)      dew   dew point, °C
    temp   temperature, °C                                      pres  pressure, hPa
    wind_dir  combined wind direction (NE / NW / SE / cv=calm)  wind_speed  cumulated wind speed, m/s
    snow_h / rain_h  cumulated hours of snow / rain
"""
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv"
PATH = Path("data/beijing_pm25.csv")
AQI_BINS = [0, 35, 75, 150, np.inf]                                   # 24-h PM2.5 (µg/m³), China AQI groups, merged to 4 classes
AQI_LABELS = ["Good", "Moderate", "Unhealthy", "Very unhealthy"]
POLLUTED = 75                                                         # "polluted day" = daily mean PM2.5 > 75 µg/m³
WEATHER = ["dew", "temp", "pres", "log_wind", "snow_h", "rain_h"]     # numeric weather inputs (no pollution info)


def _download(path=PATH):
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        print(f"downloading {URL} → {path}")
        urllib.request.urlretrieve(URL, path)
    return path


def load_hourly(path=PATH):
    """Hourly data frame indexed by timestamp."""
    raw = pd.read_csv(_download(Path(path)), na_values="NA")
    raw["datetime"] = pd.to_datetime(raw[["year", "month", "day", "hour"]])
    df = raw.rename(columns={"pm2.5": "pm25", "DEWP": "dew", "TEMP": "temp", "PRES": "pres", "cbwd": "wind_dir",
                             "Iws": "wind_speed", "Is": "snow_h", "Ir": "rain_h"})
    return df.set_index("datetime")[["pm25", "dew", "temp", "pres", "wind_dir", "wind_speed", "snow_h", "rain_h"]]


def load_daily(min_hours=12):
    """One row per day: weather summaries + daily mean PM2.5 (NaN if fewer than `min_hours` valid readings)."""
    h = load_hourly(); r = h.resample("D")
    d = pd.DataFrame({"pm25": r["pm25"].mean().where(r["pm25"].count() >= min_hours), "pm25_max": r["pm25"].max(),
                      "dew": r["dew"].mean(), "temp": r["temp"].mean(), "pres": r["pres"].mean(),
                      "temp_range": r["temp"].max() - r["temp"].min(), "pres_range": r["pres"].max() - r["pres"].min(),
                      "dew_range": r["dew"].max() - r["dew"].min(),
                      "wind_speed": r["wind_speed"].max(), "snow_h": r["snow_h"].max(), "rain_h": r["rain_h"].max(),
                      "calm_frac": r["wind_dir"].apply(lambda s: (s == "cv").mean()),
                      "wind_dir": r["wind_dir"].agg(lambda s: s.mode().iloc[0])})
    d["log_wind"] = np.log1p(d["wind_speed"])
    d["month"], d["year"], d["dayofyear"] = d.index.month, d.index.year, d.index.dayofyear
    d["season"] = d["month"].map(lambda m: "winter" if m in (12, 1, 2) else "spring" if m in (3, 4, 5) else "summer" if m in (6, 7, 8) else "autumn")
    d["log_pm25"] = np.log(d["pm25"])
    d["aqi_class"] = pd.cut(d["pm25"], AQI_BINS, labels=AQI_LABELS)
    d["polluted"] = (d["pm25"] > POLLUTED).astype(float).where(d["pm25"].notna())
    return d


def design_matrix(d, with_season=True):
    """Weather-only feature matrix (standardise later): numeric weather, wind-direction one-hot and annual-cycle terms."""
    X = d[WEATHER].copy()
    for w in ["NE", "NW", "SE"]: X[f"wind_{w}"] = (d["wind_dir"] == w).astype(float)          # 'cv' (calm) is the reference level
    if with_season:
        X["season_sin"] = np.sin(2 * np.pi * d["dayofyear"] / 365.25); X["season_cos"] = np.cos(2 * np.pi * d["dayofyear"] / 365.25)
    return X


def chrono_split(d, test_year=2014):
    """Train on 2010-2013, test on 2014 (no look-ahead)."""
    return d.index.year < test_year, d.index.year >= test_year
