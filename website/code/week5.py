#: Unit 5 – Combining models

# %% Practice 1 | Implement CART learning algorithms to perform categorization
#: Classify breast tumors as malignant or benign from measurements of cell nuclei. The Wisconsin Diagnostic Breast Cancer dataset contains real measurements and labels. CART asks a sequence of yes/no questions; use a fixed depth of 3 so its rules remain readable. Compare depths using cross-validation on training data, and evaluate the illustrative depth-3 tree once on held-out data. This classroom example is not a clinical diagnostic tool.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target.map(dict(enumerate(dataset.target_names)))
class_names = list(dataset.target_names)
feature_names = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Fit a shallow CART model and inspect its learned rules.
tree = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
tree.fit(X_train, y_train)
print(export_text(tree, feature_names=feature_names))

# Draw the actual tree so each split and its class counts can be read.
fig, axis = plt.subplots(figsize=(18, 9), layout="constrained")
nodes = plot_tree(
    tree, feature_names=feature_names, class_names=list(tree.classes_),
    filled=True, rounded=True, proportion=True, fontsize=8, ax=axis,
)
# Node boxes keep their light fill in both website themes; preserve dark text on them.
for node in nodes:
    node.set_color("black")
axis.set_title("CART decision tree (maximum depth 3)")
plt.show()

# Compare tree depths using only the training portion of the data.
depths = range(1, 13)
train_means = []
cv_means = []
cv_stds = []
for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    fold_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    train_means.append(model.score(X_train, y_train))
    cv_means.append(fold_scores.mean())
    cv_stds.append(fold_scores.std())

plt.figure(figsize=(9, 4.5), layout="constrained")
plt.plot(depths, train_means, "o-", label="training accuracy")
plt.errorbar(depths, cv_means, yerr=cv_stds, fmt="o-", capsize=3, label="5-fold CV accuracy")
plt.xlabel("maximum tree depth")
plt.ylabel("accuracy")
plt.ylim(0, 1.05)
plt.title("Tree depth: training fit and cross-validation")
plt.legend()
plt.show()

# Evaluate the depth-3 tree once on the held-out test data.
y_pred = tree.predict(X_test)
print("Held-out accuracy:", round(accuracy_score(y_test, y_pred), 3))
print(classification_report(y_test, y_pred, labels=class_names, zero_division=0))
fig, axis = plt.subplots(figsize=(6, 5), layout="constrained")
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, labels=class_names, display_labels=class_names,
    cmap="Blues", values_format="d", ax=axis,
)
axis.set_title("CART: held-out actual vs predicted diagnosis")
plt.show()

# %% Practice 2 | Implement Ensemble learning models to perform classification
#: Compare one decision tree, bagging, random forest, AdaBoost and soft voting on the same real tumor dataset. Cross-validation estimates performance using training data only. After comparing models, use each model's held-out results once. Ensembles may help, but the data decides which model performs best.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target.map(dict(enumerate(dataset.target_names)))
class_names = list(dataset.target_names)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Keep the inputs and split identical for each model.
models = {
    "Single tree": DecisionTreeClassifier(random_state=42),
    "Bagging": BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42), n_estimators=100, random_state=42,
    ),
    "Random forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
        n_estimators=50, random_state=42,
    ),
    "Voting": VotingClassifier(estimators=[
        ("tree", DecisionTreeClassifier(max_depth=3, random_state=42)),
        ("forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
    ], voting="soft"),
}

# Measure the majority-class baseline and compare models with the same training folds.
majority = y_train.value_counts().idxmax()
baseline_accuracy = (y_test == majority).mean()
print("Training-majority baseline held-out accuracy:", round(baseline_accuracy, 3))
cv_results = {}
accuracies = {}
macro_f1_scores = {}
predictions_by_model = {}
for name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring="accuracy")
    cv_results[name] = scores["test_score"]

    # Fit once on all training rows, then score held-out rows once.
    model.fit(X_train, y_train)
    heldout_predictions = model.predict(X_test)
    predictions_by_model[name] = heldout_predictions
    accuracies[name] = accuracy_score(y_test, heldout_predictions)
    macro_f1_scores[name] = f1_score(
        y_test, heldout_predictions, labels=class_names, average="macro",
    )
    print(
        name,
        "CV accuracy:", round(cv_results[name].mean(), 3),
        "+/-", round(cv_results[name].std(), 3),
        "held-out accuracy:", round(accuracies[name], 3),
        "held-out macro F1:", round(macro_f1_scores[name], 3),
    )

# Show fold-to-fold variation and held-out performance on a full accuracy scale.
names = list(models.keys())
cv_means = [cv_results[name].mean() for name in names]
cv_stds = [cv_results[name].std() for name in names]
heldout_values = [accuracies[name] for name in names]
positions = range(len(names))
plt.figure(figsize=(10, 5), layout="constrained")
plt.errorbar(positions, cv_means, yerr=cv_stds, fmt="o", capsize=4, label="5-fold CV mean +/- SD")
plt.scatter(positions, heldout_values, marker="D", label="held-out accuracy")
plt.axhline(baseline_accuracy, color="gray", linestyle="--", label="majority baseline")
plt.xticks(list(positions), names, rotation=15, ha="right")
plt.ylabel("accuracy")
plt.ylim(0, 1.05)
plt.title("Classifier comparison: training CV and held-out results")
plt.legend()
plt.show()

# Compare the held-out errors with the same malignant/benign label order.
fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
for axis, (name, heldout_predictions) in zip(axes.flat, predictions_by_model.items()):
    ConfusionMatrixDisplay.from_predictions(
        y_test, heldout_predictions, labels=class_names, display_labels=class_names,
        cmap="Blues", values_format="d", ax=axis, colorbar=False,
    )
    axis.set_title(name)
for axis in axes.flat[len(models):]:
    axis.set_visible(False)
fig.suptitle("Held-out confusion matrices (actual rows, predicted columns)")
plt.show()
