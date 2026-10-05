# Week 3 - k-means from scratch, then PCA, to find weather types
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from weather import load

d = load()
X = pd.DataFrame({"temp_max": d.temp_max, "temp_min": d.temp_min, "wind": d.wind, "log_rain": np.log1p(d.precipitation)})
Z = ((X - X.mean()) / X.std()).values            # four numbers per day, on the same scale

def kmeans(Z, k, seed, steps=100):
    centres = Z[np.random.default_rng(seed).choice(len(Z), k, replace=False)]
    for _ in range(steps):
        dist = ((Z[:, None] - centres[None]) ** 2).sum(-1)
        labels = dist.argmin(1)                                   # 1) give each day to its nearest centre
        new = np.array([Z[labels == j].mean(0) for j in range(k)])  # 2) move each centre to the mean of its days
        if np.allclose(new, centres): break
        centres = new
    return labels, dist.min(1).sum()

labels, _ = min((kmeans(Z, 4, seed) for seed in range(10)), key=lambda r: r[1])   # best of 10 starts
d["cluster"] = labels
print(d.groupby("cluster").agg(days=("rain", "size"), temp_max=("temp_max", "mean"), wind=("wind", "mean"), rain_share=("rain", "mean")).round(2), "\n")
print(pd.crosstab(d.cluster, d.weather), "\n")

pca = PCA().fit(Z)
print("variance kept by each principal component:", pca.explained_variance_ratio_.round(2))
P = pca.transform(Z)
plt.figure(figsize=(6, 4)); plt.scatter(P[:, 0], P[:, 1], c=labels, cmap="tab10", s=8)
plt.xlabel("PC1 (cool and wet <-> warm and dry)"); plt.ylabel("PC2"); plt.title("days on two axes, coloured by cluster")
plt.tight_layout(); plt.show()
