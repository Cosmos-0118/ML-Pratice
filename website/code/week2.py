# Week 2 - linear regression: predict ln PM2.5 from the weather (train 2010-13, test 2014)
import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from beijing_pm25 import load_daily, design_matrix, chrono_split

d = load_daily(); X = design_matrix(d); y = d.log_pm25
train, test = chrono_split(d)
train, test = train & y.notna(), test & y.notna()

scaler = StandardScaler().fit(X[train])
model = LinearRegression().fit(scaler.transform(X[train]), y[train])
pred = model.predict(scaler.transform(X[test]))

print("R2 on 2014:", round(r2_score(y[test], pred), 3))
print("RMSE (ln units):", round(float(np.sqrt(np.mean((y[test] - pred) ** 2))), 3))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].scatter(y[test], pred, s=10, alpha=.6); ax[0].plot([2, 6], [2, 6], "r--")
ax[0].set(xlabel="actual ln PM2.5", ylabel="predicted ln PM2.5", title="2014: predicted vs actual")
coef = dict(zip(X.columns, model.coef_)); names = sorted(coef, key=coef.get)
ax[1].barh(names, [coef[n] for n in names], color=["tab:red" if coef[n] > 0 else "tab:blue" for n in names])
ax[1].set_title("standardised coefficients")
plt.tight_layout(); plt.show()
