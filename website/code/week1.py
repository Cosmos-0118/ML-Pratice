#: Unit 1 – Introduction

# %% Practice 1 | Devise a program to import, load and view dataset
#: Read the CSV file into a table, then look at its size, its column types and its first rows.
from weather import load

d = load()                                          # reads data/seattle_weather.csv
raw = d[["date", "precipitation", "temp_max", "temp_min", "wind", "weather"]]
print("rows, columns:", raw.shape)
print("period:", d.date.min().date(), "to", d.date.max().date(), "\n")
print(raw.dtypes, "\n")
print(raw.head(10).to_string(index=False))
print("\nWeather labels:", d.weather.value_counts().to_dict())

# %% Practice 2 | Create a program to display the summary and statistics of the dataset
#: Summary statistics, missing values and skew, then histograms and the yearly cycle.
import matplotlib.pyplot as plt
from weather import load

d = load()
cols = ["precipitation", "temp_max", "temp_min", "wind"]
print(d[cols].describe().round(2), "\n")
print("missing values per column:", d[cols + ["weather"]].isna().sum().to_dict())
print("skewness:", d[cols].skew().round(2).to_dict(), "(precipitation is strongly right-skewed)")

fig, ax = plt.subplots(2, 2, figsize=(9, 5.6))
for a, c, colour in zip(ax.ravel(), cols, ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]):
    a.hist(d[c], bins=35, color=colour, alpha=.85); a.set_title(c); a.set_ylabel("days")
plt.show()

monthly = d.groupby(d.date.dt.month).agg(temp_max=("temp_max", "mean"), rain_share=("rain", "mean"))
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(monthly.index, monthly.temp_max, color="#f58518"); ax.set_title("Average temp_max by month"); ax.set_ylabel("°C")
ax.set_xlabel("month"); ax.set_xticks(range(1, 13)); plt.show()

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(monthly.index, monthly.rain_share, color="#4c78a8"); ax.set_title("Share of rainy days by month"); ax.set_ylabel("share of days")
ax.set_xlabel("month"); ax.set_xticks(range(1, 13)); plt.show()
