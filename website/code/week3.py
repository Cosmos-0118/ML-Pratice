# Week 3 - K-means weather regimes (no pollution labels used) + PCA view
import pandas as pd, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from beijing_pm25 import load_daily

d = load_daily()
cols = ["dew", "temp", "pres", "log_wind", "calm_frac"]
Z = StandardScaler().fit_transform(d[cols])

km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(Z)
d["regime"] = km.labels_
summary = d.groupby("regime").agg(days=("temp", "size"), temp=("temp", "mean"), dew=("dew", "mean"),
                                  log_wind=("log_wind", "mean"), calm=("calm_frac", "mean"),
                                  polluted=("polluted", "mean")).round(2)
print(summary.sort_values("polluted"))
print("\nThe regimes were found from weather only - 'polluted' was never used for clustering.")

P = PCA(2).fit_transform(Z)
plt.figure(figsize=(5.5, 4))
plt.scatter(P[:, 0], P[:, 1], c=km.labels_, cmap="tab10", s=8, alpha=.7)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("K-means (k=4) in PCA space")
plt.tight_layout(); plt.show()
