#: Unit 5 – Combining models

# %% Practice 1 | Implement CART learning algorithms to perform categorization
#: Classify each synthetic day as sun, cloudy, rain or snow from its measurements. CART (Classification And Regression Trees) builds a tree of yes/no questions using Gini impurity. Keep every class represented in the training and test sets, and inspect class-level results as well as overall accuracy. The deliberately distinct regimes make this an easy teaching example.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

df = pd.read_csv("data/weather_regimes.csv")
columns = ["precipitation", "temp_max", "temp_min", "wind"]
X = df[columns]
y = df["weather"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# CART with Gini impurity, at most 3 questions deep
tree = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# The learned rules. Some splits give the same class on both sides:
# they still make the groups purer (lower Gini), so CART keeps them.
print(export_text(tree, feature_names=columns))

y_pred = tree.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 3))
print(classification_report(y_test, y_pred, zero_division=0))
print()

# Classify one new day
new_day = pd.DataFrame({"precipitation": [18.0], "temp_max": [16.0], "temp_min": [12.0], "wind": [4.5]})
print("A day with 18 mm precipitation, 16 °C max, 12 °C min, wind 4.5 ->", tree.predict(new_day)[0])

# Plot: compare fit and generalization as depth grows; overfitting is not guaranteed.
depths = range(1, 13)
train_acc = []
test_acc = []
for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc.append(model.score(X_train, y_train))
    test_acc.append(model.score(X_test, y_test))

plt.figure(figsize=(9, 4.5))
plt.plot(depths, train_acc, "o-", label="training data")
plt.plot(depths, test_acc, "o-", label="test data")
plt.xlabel("tree depth")
plt.ylabel("accuracy")
plt.ylim(0, 1.05)
plt.title("Tree depth: training vs test accuracy")
plt.legend()
plt.show()

fig, axis = plt.subplots(figsize=(7, 5), layout="constrained")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=sorted(y.unique()), cmap="Blues", ax=axis)
plt.title("CART: actual vs predicted weather regime")
plt.show()

# %% Practice 2 | Implement Ensemble learning models to perform classification
#: Compare single-tree and ensemble classification on the same stratified split of synthetic data. Bagging trains trees on random samples, a random forest also samples columns, AdaBoost focuses on previous errors, and voting combines different models. An ensemble need not beat a single tree on this easy dataset; report the results without assuming which model wins.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("data/weather_regimes.csv")
columns = ["precipitation", "temp_max", "temp_min", "wind"]
X = df[columns]
y = df["weather"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit each model on the same inputs. Labels are targets, never input columns.
models = {
    "Single tree (no depth limit)": DecisionTreeClassifier(random_state=42),
    "Bagging": BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=42),
    "Random forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=50, random_state=42),
    "Voting": VotingClassifier([
        ("tree", DecisionTreeClassifier(max_depth=3, random_state=42)),
        ("forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
    ]),
}

accuracies = {}
majority = y_train.value_counts().idxmax()
print("Training-majority baseline accuracy:", round((y_test == majority).mean(), 3))
for name in models:
    model = models[name]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    print(name, "accuracy:", round(accuracies[name], 3), "macro F1:", round(macro_f1, 3))

# Plot: compare the accuracies
plt.figure(figsize=(9, 4.5), layout="constrained")
bars = plt.barh(list(accuracies.keys()), list(accuracies.values()))
plt.bar_label(bars, fmt="%.3f", padding=3)
plt.xlim(0, 1.1)
plt.xlabel("accuracy")
plt.title("Single tree vs ensembles")
plt.show()
