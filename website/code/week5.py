# Week 5 - single tree vs ensembles vs logistic regression (train 2010-13, test 2014)
import pandas as pd, matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from beijing_pm25 import load_daily, context_features, chrono_split

d = load_daily(); X = context_features(d); y = d.polluted
ok = y.notna() & X.notna().all(axis=1)
train, test = chrono_split(d); train, test = train & ok, test & ok

models = {
    "Single tree (depth 5)": DecisionTreeClassifier(max_depth=5, random_state=0),
    "Random forest":         RandomForestClassifier(200, min_samples_leaf=5, random_state=0),
    "Gradient boosting":     HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=150, random_state=0),
    "Logistic regression":   make_pipeline(StandardScaler(), LogisticRegression(C=.3, max_iter=2000)),
}
rows = []
for name, m in models.items():
    m.fit(X[train], y[train]); p = m.predict_proba(X[test])[:, 1]
    rows.append((name, accuracy_score(y[test], p > .5), roc_auc_score(y[test], p)))
res = pd.DataFrame(rows, columns=["model", "accuracy", "AUC"]).set_index("model").round(3)
print(res)

res.plot.barh(figsize=(8, 3.4)); plt.xlim(.6, 1); plt.title("Polluted day? 2014 test year")
plt.tight_layout(); plt.show()
