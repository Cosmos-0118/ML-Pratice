# Week 2 - linear regression (tomorrow's temperature) and logistic regression (rain tomorrow)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from weather import load, split, FEATURES

train, test = split(load())              # train on 2012-2014, test on 2015

# --- Regression: least squares with the normal equation w = (X'X)^-1 X'y
X_tr = np.c_[np.ones(len(train)), train[FEATURES]]
X_te = np.c_[np.ones(len(test)), test[FEATURES]]
w = np.linalg.solve(X_tr.T @ X_tr, X_tr.T @ train.temp_tomorrow)
pred = X_te @ w
rmse = lambda p: np.sqrt(np.mean((test.temp_tomorrow - p) ** 2))
print(f"tomorrow's temp_max, error in C:  'same as today' {rmse(test.temp_max):.2f}  |  linear regression {rmse(pred):.2f}")

# --- Classification: will it rain tomorrow?
scaler = StandardScaler().fit(train[FEATURES])
model = LogisticRegression().fit(scaler.transform(train[FEATURES]), train.rain_tomorrow)
accuracy = model.score(scaler.transform(test[FEATURES]), test.rain_tomorrow)
print(f"rain tomorrow: 'always no rain' {1 - test.rain_tomorrow.mean():.3f}  |  'same as today' {(test.rain == test.rain_tomorrow).mean():.3f}  |  logistic regression {accuracy:.3f}")

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.scatter(test.temp_tomorrow, pred, s=8, alpha=.6); ax.plot([0, 35], [0, 35], "r--")
ax.set_xlabel("actual temp_max tomorrow (C)"); ax.set_ylabel("predicted"); ax.set_title("linear regression on 2015")
plt.tight_layout(); plt.show()
