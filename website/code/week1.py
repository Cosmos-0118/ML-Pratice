# Week 1 - load the data, summarise it, look at the seasons
import numpy as np
import matplotlib.pyplot as plt
from weather import load

d = load()
print("days:", len(d), "| first:", d.date.min().date(), "| last:", d.date.max().date())
print("missing values:", int(d[["precipitation", "temp_max", "temp_min", "wind"]].isna().sum().sum()))
print(d[["precipitation", "temp_max", "temp_min", "wind"]].describe().round(1), "\n")

print("skew of precipitation:", round(d.precipitation.skew(), 2), "(very skewed: mostly dry days, a few very wet ones)")

# Bayes' rule: how much does rain today change the chance of rain tomorrow?
t = d.dropna()
print("P(rain tomorrow)              =", round(t.rain_tomorrow.mean(), 2))
print("P(rain tomorrow | rain today) =", round(t[t.rain == 1].rain_tomorrow.mean(), 2))
print("P(rain tomorrow | dry today)  =", round(t[t.rain == 0].rain_tomorrow.mean(), 2))

by_month = d.groupby(d.date.dt.month).agg(temp_max=("temp_max", "mean"), rain_share=("rain", "mean"))
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
by_month.temp_max.plot.bar(ax=ax[0], color="tab:orange", title="average temp_max (C) by month")
by_month.rain_share.plot.bar(ax=ax[1], color="tab:blue", title="share of rainy days by month")
plt.tight_layout(); plt.show()
