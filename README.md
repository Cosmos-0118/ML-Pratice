# Machine Learning Practice (21CSC305P)

One Jupyter notebook per week/unit of the **21CSC305P – Machine Learning** syllabus. Each notebook has
**Part A** (theory in code: small experiments that illustrate the unit's topics) and **Part B** (the unit's *Practice* tasks, implemented and evaluated).
All notebooks are saved **with their outputs and graphs**, so they can be read directly on GitHub, and they re-run top-to-bottom with no downloads
(only scikit-learn's bundled datasets are used).

| Week | Notebook | Unit | Practice tasks covered |
|---|---|---|---|
| 1 | [`Week_01_Introduction_to_ML.ipynb`](Week_01_Introduction_to_ML.ipynb) | Introduction | 1. Import, load and view a dataset · 2. Summary & statistics of a dataset |
| 2 | [`Week_02_Linear_Models.ipynb`](Week_02_Linear_Models.ipynb) | Linear models for regression & classification | 1. Linear regression for prediction · 2. Bayesian logistic regression and SVM |
| 3 | [`Week_03_Clustering_EM_PCA.ipynb`](Week_03_Clustering_EM_PCA.ipynb) | Mixture models and EM | 1. K-means, mixtures of Gaussians, hierarchical clustering · 2. PCA |
| 4 | [`Week_04_Hidden_Markov_Models.ipynb`](Week_04_Hidden_Markov_Models.ipynb) | Hidden Markov models | 1. HMM to predict sequential data |
| 5 | [`Week_05_Combining_Models_Ensembles.ipynb`](Week_05_Combining_Models_Ensembles.ipynb) | Combining models | 1. CART for categorisation · 2. Ensemble learning for classification |

## Highlights (implemented from scratch with NumPy, then validated against scikit-learn / brute force)
* Week 1 – polynomial curve fitting & over-fitting, Bayes' rule, (conditional) independence, quantiles, covariance; reusable `summarize()` EDA function.
* Week 2 – ML/ridge/robust/Bayesian linear regression, **Bayesian logistic regression (Laplace approximation)**, kernel trick, SVM tuning, kernelised logistic regression.
* Week 3 – **K-means (k-means++)**, **EM for Gaussian mixtures**, hierarchical clustering, silhouette/ARI/NMI, **PCA (eigen + SVD)**, factor analysis, choosing the latent dimension.
* Week 4 – **HMM** with scaled forward–backward, **Baum–Welch**, **Viterbi**, next-step forecasting, Gaussian-emission regime detection, **Kalman filter / RTS smoother**.
* Week 5 – Bayesian model averaging, bagging, **AdaBoost**, **gradient boosting**, GAM, **CART (Gini/entropy, pruning)**, bagging/OOB, voting, stacking, consensus clustering.

## Run it
```bash
pip install -r requirements.txt
jupyter lab        # open any Week_XX notebook and Run All
```
