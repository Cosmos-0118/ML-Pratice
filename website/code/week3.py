#: Unit 3 – Mixture models and EM

# %% Practice 1 | Implement K-means clustering, mixtures of Gaussians and Hierarchical clustering algorithm to categorize data
#: Use the real UCI Wine dataset: 178 wines described by 13 chemical measurements. Standardize the measurements and fit every clustering method in all 13 dimensions. Choose K-means k using silhouette and GMM complexity using BIC; cut the Ward tree at the selected K-means count for comparison. The common PCA view is only a projection. Each method also has its own diagnostic plots. Known cultivars are used only to evaluate agreement after fitting, never to choose the models.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import chi2
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score

dataset = load_wine(as_frame=True)
columns = dataset.feature_names
ink = plt.rcParams["text.color"]  # follows the website theme and stays black in notebooks
X = StandardScaler().fit_transform(dataset.data)

# K-means: compare candidate counts without using the cultivar labels.
counts = np.arange(2, 7)
kmeans_candidates = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X) for k in counts]
inertias = [model.inertia_ for model in kmeans_candidates]
silhouettes = [silhouette_score(X, model.labels_) for model in kmeans_candidates]
kmeans = kmeans_candidates[np.argmax(silhouettes)]
k = kmeans.n_clusters

# GMM: BIC trades fit against model complexity. Lower is better.
gmm_counts = np.arange(1, 7)
covariance_types = ["tied", "diag", "full"]
gmm_candidates = {}
for covariance_type in covariance_types:
    for count in gmm_counts:
        model = GaussianMixture(n_components=count, covariance_type=covariance_type,
                                n_init=3, reg_covar=1e-4, max_iter=300, random_state=42).fit(X)
        if not model.converged_:
            raise RuntimeError("GMM did not converge; inspect the candidate before comparing BIC.")
        gmm_candidates[covariance_type, count] = model
gmm = min(gmm_candidates.values(), key=lambda model: model.bic(X))
responsibilities = gmm.predict_proba(X)

# Use this exact tree for both the Ward assignments and the dendrogram.
tree = linkage(X, method="ward")
cut_height = (tree[-k, 2] + tree[-(k - 1), 2]) / 2
ward_labels = fcluster(tree, cut_height, criterion="distance") - 1
labels_by_method = {"K-means": kmeans.labels_, "GMM": gmm.predict(X), "Ward": ward_labels}
print("Wine samples:", X.shape[0], "| standardized measurements:", X.shape[1])
print("K-means chosen k:", k, "| GMM components:", gmm.n_components, "| covariance:", gmm.covariance_type)
print("Scores use all 13 features. Cultivar agreement is ARI, not classification accuracy.")
for name, labels in labels_by_method.items():
    print(name, "sizes:", np.bincount(labels).tolist(),
          "silhouette:", round(silhouette_score(X, labels), 3),
          "cultivar ARI:", round(adjusted_rand_score(dataset.target, labels), 3))
print("GMM component probabilities for the first five wines:")
print(responsibilities[:5].round(3))

# A single projection keeps the comparison fair; it does not determine clusters.
projection = PCA(n_components=2).fit(X)
points = projection.transform(X)
variance = projection.explained_variance_ratio_ * 100

def component_covariance(component):
    if gmm.covariance_type == "full":
        return gmm.covariances_[component]
    if gmm.covariance_type == "tied":
        return gmm.covariances_
    if gmm.covariance_type == "diag":
        return np.diag(gmm.covariances_[component])
    return np.eye(X.shape[1]) * gmm.covariances_[component]

# Covariances must be projected too: P @ covariance @ P.T.
P = projection.components_
projected_covariances = np.array([P @ component_covariance(i) @ P.T for i in range(gmm.n_components)])
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True, sharey=True, layout="constrained")
for ax, (name, labels) in zip(axes, labels_by_method.items()):
    for cluster in np.unique(labels):
        members = labels == cluster
        ax.scatter(points[members, 0], points[members, 1], s=20, alpha=0.75, label=f"Group {cluster + 1}")
    ax.set_title(name)
    ax.set_xlabel(f"PC1 ({variance[0]:.1f}%)")
    ax.legend(fontsize=8)
axes[0].set_ylabel(f"PC2 ({variance[1]:.1f}%)")
centres = projection.transform(kmeans.cluster_centers_)
axes[0].scatter(centres[:, 0], centres[:, 1], marker="X", s=140, c=ink, label="Centroids")
axes[0].legend(fontsize=8)
for centre, covariance in zip(projection.transform(gmm.means_), projected_covariances):
    values, vectors = np.linalg.eigh(covariance)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    width, height = 2 * np.sqrt(chi2.ppf(0.95, 2) * values[::-1])
    axes[1].add_patch(Ellipse(centre, width, height, angle=angle, fill=False, edgecolor=ink, linewidth=1.2))
fig.suptitle(f"Shared PCA view: {variance.sum():.1f}% variance retained; GMM contours contain 95% of each projected Gaussian")
plt.show()

# K-means diagnostics: elbow and silhouette describe different properties.
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
axes[0].plot(counts, inertias, "o-")
axes[0].set_ylabel("Within-cluster sum of squares")
axes[0].set_title("K-means elbow")
axes[1].plot(counts, silhouettes, "o-")
axes[1].set_ylabel("Mean silhouette (all 13 features)")
axes[1].set_title("K-means separation")
for ax in axes:
    ax.axvline(k, color=ink, linestyle="--", label=f"Selected k = {k}")
    ax.set_xlabel("Number of clusters")
    ax.set_xticks(counts)
    ax.legend()
plt.show()

samples = silhouette_samples(X, kmeans.labels_)
fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
bottom = 0
for cluster in range(k):
    values = np.sort(samples[kmeans.labels_ == cluster])
    ax.fill_betweenx(np.arange(bottom, bottom + len(values)), 0, values, alpha=0.7, label=f"Group {cluster + 1} (n={len(values)})")
    bottom += len(values) + 6
ax.axvline(samples.mean(), color=ink, linestyle="--", label=f"Mean = {samples.mean():.3f}")
ax.set(xlabel="Silhouette coefficient (negative values suggest overlap)", ylabel="Samples grouped by cluster", title="K-means silhouette profiles", xlim=(-1, 1), yticks=[])
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
for covariance_type in covariance_types:
    ax.plot(gmm_counts, [gmm_candidates[covariance_type, count].bic(X) for count in gmm_counts], "o-", label=covariance_type)
ax.scatter([gmm.n_components], [gmm.bic(X)], marker="*", s=180, c=ink, label="Selected model")
ax.set(xlabel="Number of mixture components", ylabel="BIC (lower is better)", title="GMM: component count and covariance selection", xticks=gmm_counts)
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
scatter = ax.scatter(points[:, 0], points[:, 1], c=responsibilities.max(axis=1), vmin=0, vmax=1, cmap="viridis", s=30)
fig.colorbar(scatter, ax=ax, label="Largest fitted component probability")
ax.set(xlabel=f"PC1 ({variance[0]:.1f}%)", ylabel=f"PC2 ({variance[1]:.1f}%)", title="GMM soft assignments: confidence under the fitted model")
plt.show()

fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
dendrogram(tree, truncate_mode="lastp", p=20, color_threshold=cut_height, above_threshold_color="grey", ax=ax)
ax.axhline(cut_height, color=ink, linestyle="--", label=f"Cut for {k} groups")
ax.set(xlabel="Merged groups (parentheses give sample counts)", ylabel="Ward linkage distance", title="Hierarchical clustering: the tree used for assignments")
ax.legend()
plt.show()

profiles = pd.DataFrame(X, columns=columns).groupby(ward_labels).mean()
fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
limit = np.abs(profiles.values).max()
heatmap = ax.imshow(profiles, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
ax.set_xticks(range(len(columns)), labels=[name.replace("_", " ") for name in columns], rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(len(profiles)), labels=[f"Group {i + 1}" for i in profiles.index])
ax.set_title("Ward groups: mean standardized chemical measurements")
fig.colorbar(heatmap, ax=ax, label="Mean standardized value")
plt.show()

# %% Practice 2 | Create a program to perform PCA
#: Perform PCA on all 13 standardized chemical measurements in the real Wine dataset. Inspect correlations, the full scree plot, cumulative variance, three score views, correlation loadings and a symmetric SVD biplot. Cultivars colour the score plots only. Two dimensions are a visual summary; the 90% and 95% thresholds show how many components are needed to retain more information. Component signs are arbitrary.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataset = load_wine(as_frame=True)
columns = dataset.feature_names
ink = plt.rcParams["text.color"]  # follows the website theme and stays black in notebooks
names = [name.replace("_", " ") for name in columns]
X = StandardScaler().fit_transform(dataset.data)
pca = PCA()
X_pca = pca.fit_transform(X)
variance = pca.explained_variance_ratio_
cumulative = np.cumsum(variance)
k90 = np.searchsorted(cumulative, 0.90) + 1
k95 = np.searchsorted(cumulative, 0.95) + 1
pcs = np.arange(1, len(columns) + 1)
print("Explained variance by component:")
print(pd.DataFrame({"component": pcs, "variance %": variance * 100, "cumulative %": cumulative * 100}).round(2).to_string(index=False))
print(f"First two PCs keep {100 * cumulative[1]:.1f}% of variance; 90% needs {k90} PCs and 95% needs {k95} PCs.")

# Correlation loadings are correlations of original standardized features with scores.
# The sample standard deviation accounts for PCA's n-1 variance convention.
correlation_loadings = pca.components_.T * np.sqrt(pca.explained_variance_) / X.std(axis=0, ddof=1)[:, None]
print("Feature correlations with the first three PC scores:")
print(pd.DataFrame(correlation_loadings[:, :3], index=columns, columns=["PC1", "PC2", "PC3"]).round(3))

# 1. Input correlations: which measurements contain related information?
fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
heatmap = ax.imshow(np.corrcoef(X.T), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(names)), labels=names, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(names)), labels=names, fontsize=8)
ax.set_title("Wine: correlations between the 13 input measurements")
fig.colorbar(heatmap, ax=ax, label="Pearson correlation")
plt.show()

# 2. Scree: show every component, not just the first two.
fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
ax.bar(pcs, variance * 100)
ax.set(xlabel="Principal component", ylabel="Explained variance (%)", title="Scree plot: all 13 principal components", xticks=pcs)
plt.show()

# 3. Cumulative variance: mark both thresholds and their first crossings.
fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
ax.plot(pcs, cumulative * 100, "o-")
for threshold, count, colour in [(90, k90, "#d97706"), (95, k95, "#7c3aed")]:
    ax.axhline(threshold, color=colour, linestyle="--", label=f"{threshold}% requires {count} PCs")
    ax.axvline(count, color=colour, linestyle=":")
ax.set(xlabel="Number of retained components", ylabel="Cumulative explained variance (%)", title="Choosing how many components to keep", xticks=pcs, ylim=(0, 105))
ax.legend()
plt.show()

# 4. Score views: PC3 can reveal structure lost in the PC1-PC2 view.
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
for ax, (first, second) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
    for cultivar, name in enumerate(dataset.target_names):
        members = dataset.target == cultivar
        ax.scatter(X_pca[members, first], X_pca[members, second], s=24, alpha=0.75, label=name)
    ax.set_xlabel(f"PC{first + 1} ({100 * variance[first]:.1f}%)")
    ax.set_ylabel(f"PC{second + 1} ({100 * variance[second]:.1f}%)")
    ax.legend(fontsize=8)
fig.suptitle("PCA score views; known cultivars are colours, not PCA inputs")
plt.show()

# 5. Full feature interpretation, using correlations rather than raw eigenvectors.
fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
heatmap = ax.imshow(correlation_loadings[:, :3], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(3), labels=["PC1", "PC2", "PC3"])
ax.set_yticks(range(len(names)), labels=names, fontsize=9)
for row in range(len(names)):
    for col in range(3):
        value = correlation_loadings[row, col]
        ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white" if abs(value) > 0.65 else "black", fontsize=9)
ax.set_title("Correlation loadings: every feature against PC1-PC3")
fig.colorbar(heatmap, ax=ax, label="Correlation with component score")
plt.show()

# 6. Symmetric SVD biplot: the product reconstructs the two-PC approximation.
# X_centered = U S V.T; use U sqrt(S) and V sqrt(S), without arbitrary arrow magnification.
root_singular = np.sqrt(pca.singular_values_[:2])
biplot_samples = X_pca[:, :2] / root_singular
biplot_features = pca.components_[:2].T * root_singular
# Only the six features best represented in this plane get arrows; the heatmap shows all 13.
shown = np.argsort((correlation_loadings[:, :2] ** 2).sum(axis=1))[-6:]
fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
for cultivar, name in enumerate(dataset.target_names):
    members = dataset.target == cultivar
    ax.scatter(biplot_samples[members, 0], biplot_samples[members, 1], s=24, alpha=0.6, label=name)
for i, feature in enumerate(shown):
    endpoint = biplot_features[feature]
    ax.annotate("", xy=endpoint, xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": ink, "lw": 1.4})
    ax.annotate(names[feature], xy=endpoint, xytext=(8, 10 if i % 2 == 0 else -16), textcoords="offset points", fontsize=9)
ax.axhline(0, color="grey", linewidth=0.6)
ax.axvline(0, color="grey", linewidth=0.6)
extents = np.vstack([biplot_samples, biplot_features[shown]])
ax.set_xlim(extents[:, 0].min() - 0.5, extents[:, 0].max() + 1.1)
ax.set_ylim(extents[:, 1].min() - 0.5, extents[:, 1].max() + 0.5)
ax.set_aspect("equal", adjustable="box")
ax.set(xlabel=f"PC1 ({100 * variance[0]:.1f}%), symmetric SVD scale", ylabel=f"PC2 ({100 * variance[1]:.1f}%), symmetric SVD scale", title="Wine biplot: samples and six best-represented feature directions")
ax.legend()
plt.show()
