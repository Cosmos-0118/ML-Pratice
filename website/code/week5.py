#: Unit 5 – Combining models

# %% Practice 1 | Implement CART learning algorithms to perform categorization
#: Use real Wisconsin breast cancer measurements to train a decision tree. A depth of 3 keeps the rules readable. Compare tree depths using training cross-validation, then inspect errors on a separate test set. This is a classroom example.
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# 1. Load features and labels: 0 = malignant, 1 = benign.
dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target
class_names = dataset.target_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Train CART. The test rows are not used during training.
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print("Labels: 0 = malignant, 1 = benign")
print(export_text(tree, feature_names=list(X.columns)))
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 3))
print(classification_report(y_test, y_pred, target_names=class_names))

# Graph 1: the actual questions asked by the tree.
plt.figure(figsize=(16, 8))
nodes = plot_tree(tree, feature_names=X.columns, class_names=class_names, filled=True, rounded=True, fontsize=9)
for node in nodes:
    node.set_color("black")  # readable text on the coloured boxes in both website themes
plt.title("CART decision tree: maximum depth 3")
plt.show()

# Graph 2: more depth can improve training fit without improving new predictions.
depths = range(1, 7)
training_scores = []
validation_scores = []
for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    training_scores.append(model.score(X_train, y_train))
    fold_scores = cross_val_score(model, X_train, y_train, cv=cv)
    validation_scores.append(fold_scores.mean())
plt.figure(figsize=(9, 5))
plt.plot(depths, training_scores, "o-", label="Training accuracy")
plt.plot(depths, validation_scores, "o-", label="Mean of 5 training CV folds")
plt.xlabel("Maximum tree depth")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.title("Tree depth and accuracy")
plt.legend()
plt.show()

# Graph 3: correct predictions are on the diagonal; errors are off the diagonal.
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=[0, 1], display_labels=class_names, cmap="Blues")
plt.title("CART: actual rows, predicted columns")
plt.show()

# %% Practice 2 | Implement Ensemble learning models to perform classification
#: Compare a single tree, bagging, random forest, AdaBoost and majority voting on the same real dataset and split. Use only training rows for cross-validation. Show held-out accuracy and errors; an ensemble does not have to beat every other model.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

# 1. Use the same real features, labels and split as the CART exercise.
dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target
class_names = dataset.target_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Create the models. Their library classes do the algorithm's internal work.
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
bagging = BaggingClassifier(n_estimators=50, random_state=42)  # trees trained on resampled rows
forest = RandomForestClassifier(n_estimators=50, random_state=42)  # trees also try random feature subsets
boost = AdaBoostClassifier(n_estimators=50, random_state=42)  # later models focus on earlier errors
voting = VotingClassifier([("tree", tree), ("forest", forest), ("boost", boost)], voting="hard")
models = {"Tree": tree, "Bagging": bagging, "Forest": forest, "AdaBoost": boost, "Voting": voting}

# 3. Compare models using the same folds and the same held-out test rows.
cv_results = {}
accuracies = {}
predictions = {}
cv_means = []
test_scores = []
for name, model in models.items():
    fold_scores = cross_val_score(model, X_train, y_train, cv=cv)
    cv_results[name] = fold_scores
    cv_means.append(fold_scores.mean())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    accuracies[name] = accuracy_score(y_test, y_pred)
    test_scores.append(accuracies[name])
    print(name, "CV accuracy:", round(fold_scores.mean(), 3),
          "test accuracy:", round(accuracies[name], 3),
          "macro F1:", round(f1_score(y_test, y_pred, average="macro"), 3))
majority_class = y_train.value_counts().idxmax()
baseline = (y_test == majority_class).mean()
print("Always predict the training majority:", round(baseline, 3))

# Graph 1: two ordinary bars per model.
positions = np.arange(len(models))
plt.figure(figsize=(10, 5))
plt.bar(positions - 0.2, cv_means, width=0.4, label="Mean training CV accuracy")
plt.bar(positions + 0.2, test_scores, width=0.4, label="Test accuracy")
plt.axhline(baseline, color="grey", linestyle="--", label="Majority baseline")
plt.xticks(positions, list(models))
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.title("Single tree and ensemble comparison")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.show()

# Graph 2: the same simple confusion matrix for each model.
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, name in enumerate(models):
    axis = axes.flat[i]  # take the next drawing box in the grid
    ConfusionMatrixDisplay.from_predictions(y_test, predictions[name], labels=[0, 1],
        display_labels=class_names, cmap="Blues", colorbar=False, ax=axis)
    axis.set_title(name)
axes.flat[5].set_visible(False)  # five models need five boxes
fig.suptitle("Test errors: actual rows, predicted columns")
plt.show()
