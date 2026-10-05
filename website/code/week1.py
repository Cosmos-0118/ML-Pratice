# Week 1 - load the data, summarise it, look at its shape
import matplotlib.pyplot as plt
from beijing_pm25 import load_hourly, load_daily

hourly = load_hourly()
print("hourly rows:", len(hourly), "| hours with no PM2.5 reading:", int(hourly.pm25.isna().sum()))
print(hourly.describe().round(1).T[["mean", "std", "min", "50%", "max"]], "\n")

daily = load_daily()
print("days with a daily mean:", int(daily.pm25.notna().sum()))
print("share of polluted days (> 75 ug/m3):", round(daily.polluted.mean(), 3))
print("skew of PM2.5:", round(daily.pm25.skew(), 2), "-> skew of ln PM2.5:", round(daily.log_pm25.skew(), 2))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
daily.pm25.hist(bins=50, ax=ax[0], color="tab:red");      ax[0].set_title("daily PM2.5 (ug/m3): skewed")
daily.log_pm25.hist(bins=50, ax=ax[1], color="tab:blue"); ax[1].set_title("ln PM2.5: close to normal")
plt.tight_layout(); plt.show()
