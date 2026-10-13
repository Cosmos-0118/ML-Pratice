#: Unit 3 – Mixture models and EM

# %% Practice 1 | Implement K-means clustering, mixtures of Gaussians and Hierarchical clustering algorithm to categorize data
#: Group real Wine measurements using three clustering methods. Fit each model using all 13 standardized features. Use three groups as a simple teaching choice; the elbow plot helps inspect that choice. PCA only provides coordinates for drawing the wines. Cluster numbers are arbitrary.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# 1. Load real measurements and put the features on the same scale.
dataset = load_wine(as_frame=True)
columns = dataset.feature_names
X = StandardScaler().fit_transform(dataset.data)

# 2. Fit K-means. Each wine joins the group with the nearest centre.
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# 3. Fit a Gaussian mixture. Each wine gets a probability for every group.
gmm = GaussianMixture(n_components=3, n_init=3, random_state=42)
gmm_labels = gmm.fit_predict(X)
responsibilities = gmm.predict_proba(X)
confidence = responsibilities.max(axis=1)  # largest probability in each row
print("GMM probabilities for the first five wines:")
print(responsibilities[:5].round(2))

# 4. Hierarchical clustering joins nearby groups into a tree.
tree = linkage(X, method="ward")
ward_labels = fcluster(tree, 3, criterion="maxclust") - 1
labels_by_method = {"K-means": kmeans_labels, "GMM": gmm_labels, "Ward": ward_labels}
for name, labels in labels_by_method.items():
    print(name, "group sizes:", np.bincount(labels))
    print("Silhouette:", round(silhouette_score(X, labels), 3), "(higher means less overlap)")

# 5. Create two drawing coordinates. Models above still use all 13 features.
projection = PCA(n_components=2)
points = projection.fit_transform(X)
centres = projection.transform(kmeans.cluster_centers_)
print("Variance kept in the drawing:", round(projection.explained_variance_ratio_.sum() * 100, 1), "%")
ink = plt.rcParams["text.color"]

# Graph 1: K-means groups and their centres.
plt.figure(figsize=(9, 5))
for group in range(3):
    members = kmeans_labels == group
    plt.scatter(points[members, 0], points[members, 1], label="Group " + str(group + 1), s=25)
plt.scatter(centres[:, 0], centres[:, 1], marker="X", s=160, color=ink, label="Centres")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-means: three groups in a PCA view")
plt.legend()
plt.show()

# Graph 2: GMM confidence, rather than repeating the K-means graph.
plt.figure(figsize=(9, 5))
plt.scatter(points[:, 0], points[:, 1], c=confidence, cmap="viridis", vmin=0, vmax=1, s=30)
plt.colorbar(label="Largest group probability")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Gaussian mixture: confidence in each wine's group")
plt.show()

# Graph 3: the actual hierarchy used to form the Ward groups.
plt.figure(figsize=(10, 5))
dendrogram(tree, truncate_mode="lastp", p=20)
plt.xlabel("Merged groups (parentheses show sample counts)")
plt.ylabel("Ward joining distance")
plt.title("Hierarchical clustering: dendrogram")
plt.show()

# Graph 4: inspect the K-means choice with a simple elbow plot.
counts = range(1, 7)
inertias = []
for count in counts:
    model = KMeans(n_clusters=count, n_init=10, random_state=42)
    model.fit(X)
    inertias.append(model.inertia_)  # total squared distance from group centres
plt.figure(figsize=(9, 5))
plt.plot(counts, inertias, "o-")
plt.xlabel("Number of groups")
plt.ylabel("Within-group squared distance")
plt.title("K-means elbow: look for where the decrease becomes smaller")
plt.xticks(counts)
plt.show()

# %% Practice 2 | Create a program to perform PCA
#: PCA combines the 13 Wine measurements into new features called principal components. These graphs show each component's variance, cumulative variance, the first two scores, and the feature weights. Cultivar labels only colour the scatter plot; PCA does not use them.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load and standardize the real measurements.
dataset = load_wine(as_frame=True)
columns = dataset.feature_names
X = StandardScaler().fit_transform(dataset.data)

# 2. Calculate all principal components.
pca = PCA()
X_pca = pca.fit_transform(X)
variance = pca.explained_variance_ratio_
cumulative = np.cumsum(variance)
components = range(1, len(columns) + 1)
print("Variance explained (%):", (variance * 100).round(2))
print("First two components keep:", round(cumulative[1] * 100, 1), "%")
k95 = 1
while cumulative[k95 - 1] < 0.95:
    k95 += 1
print("Components needed for 95%:", k95)

# Graph 1: how much information each component keeps (scree plot).
plt.figure(figsize=(9, 5))
plt.bar(components, variance * 100)
plt.xlabel("Principal component")
plt.ylabel("Explained variance (%)")
plt.title("PCA scree plot: all 13 components")
plt.xticks(components)
plt.show()

# Graph 2: how much information several components keep together.
plt.figure(figsize=(9, 5))
plt.plot(components, cumulative * 100, "o-")
plt.axhline(95, color="red", linestyle="--", label="95% of variance")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance (%)")
plt.title("PCA cumulative variance")
plt.xticks(components)
plt.ylim(0, 105)
plt.legend()
plt.show()

# Graph 3: wines drawn on PC1 and PC2; colours are the known cultivars.
plt.figure(figsize=(9, 5))
for cultivar in range(3):
    members = dataset.target == cultivar
    plt.scatter(X_pca[members, 0], X_pca[members, 1], s=25, label="Cultivar " + str(cultivar + 1))
plt.xlabel("PC1 (" + str(round(variance[0] * 100, 1)) + "%)")
plt.ylabel("PC2 (" + str(round(variance[1] * 100, 1)) + "%)")
plt.title("Wine PCA scores: a two-dimensional summary")
plt.legend()
plt.show()

# Graph 4: weights tell us which features form PC1 and PC2.
# .T swaps rows and columns, giving one row per original feature.
loadings = pd.DataFrame(pca.components_[:2].T, index=columns, columns=["PC1", "PC2"])
print("Feature weights (not feature-to-score correlations):")
print(loadings.round(3))
loadings.plot(kind="barh", figsize=(10, 7))
plt.xlabel("Feature weight")
plt.ylabel("Original feature")
plt.title("PCA loadings: feature weights in the first two components")
plt.show()
