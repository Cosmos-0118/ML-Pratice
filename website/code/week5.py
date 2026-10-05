# Week 5 - a single tree, ensembles of trees, and a plain logistic regression: who predicts rain best?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from weather import load, split, FEATURES

train, test = split(load())              # train on 2012-2014, test on 2015
X_tr, y_tr, X_te, y_te = train[FEATURES], train.rain_tomorrow, test[FEATURES], test.rain_tomorrow

models = {
    "single deep tree": DecisionTreeClassifier(min_samples_leaf=10, random_state=0),
    "bagging (100 trees)": BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=10), n_estimators=100, random_state=0),
    "random forest": RandomForestClassifier(300, min_samples_leaf=5, random_state=0),
    "gradient boosting": GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=.05, random_state=0),
    "logistic regression": LogisticRegression(max_iter=1000),
}
baseline = (test.rain == y_te).mean()
scores = pd.Series({name: m.fit(X_tr, y_tr).score(X_te, y_te) for name, m in models.items()})
print(f"'tomorrow = today' accuracy on 2015: {baseline:.3f}\n")
print(scores.round(3).sort_values(ascending=False))

scores.sort_values().plot.barh(color="tab:blue"); plt.axvline(baseline, color="r", ls="--", label="tomorrow = today")
plt.xlim(.6, .8); plt.xlabel("accuracy on 2015"); plt.legend(); plt.title("Will it rain tomorrow?")
plt.tight_layout(); plt.show()
