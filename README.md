# Machine Learning Practice (21CSC305P)

One Jupyter notebook per week/unit of the **21CSC305P – Machine Learning** syllabus. Each notebook has
**Part A** (theory in code: small experiments that illustrate the unit's topics) and **Part B** (the unit's *Practice* tasks, implemented and evaluated).
All notebooks are saved **with their outputs and graphs**, so they can be read directly on GitHub, and they re-run top-to-bottom.

## One dataset for the whole course: Beijing PM2.5 air quality
Hourly PM2.5 (µg/m³) plus weather — dew point, temperature, pressure, wind direction, cumulated wind speed, snow / rain hours — for **Beijing, 1 Jan 2010 – 31 Dec 2014** (43,824 hours, 2,067 missing PM2.5 readings).

* Source: UCI Machine Learning Repository, [*Beijing PM2.5 Data*](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data) (licence **CC BY 4.0**).
  Liang, X. et al. (2015), *Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating*, Proc. R. Soc. A 471: 20150257.
  PM2.5 from the US Embassy in Beijing, weather from Beijing Capital International Airport.
* The CSV is kept in [`data/beijing_pm25.csv`](data/beijing_pm25.csv) (a copy of the UCI file, taken from the mirror <https://github.com/jbrownlee/Datasets>); [`beijing_pm25.py`](beijing_pm25.py) loads it (and re-downloads it if it is missing) and builds the daily table and labels used by every notebook.
* Daily table: 1,826 days (1,753 with ≥ 12 valid PM2.5 hours). Labels: 4-class air quality (Good ≤ 35 · Moderate ≤ 75 · Unhealthy ≤ 150 · Very unhealthy > 150 µg/m³) and **polluted day** = PM2.5 > 75 µg/m³ (52.6 % of days).
* Evaluation is **chronological**: train on 2010–2013, test on 2014, cross-validation with expanding-window time-series splits.

| Week | Notebook | Unit | Practice tasks covered — on the Beijing data |
|---|---|---|---|
| 1 | [`Week_01_Introduction_to_ML.ipynb`](Week_01_Introduction_to_ML.ipynb) | Introduction | 1. Import, load and view the dataset · 2. Summary & statistics (missing data, skew, seasonality, daily table) |
| 2 | [`Week_02_Linear_Models.ipynb`](Week_02_Linear_Models.ipynb) | Linear models for regression & classification | 1. Linear regression: predict ln PM2.5 from weather · 2. Bayesian logistic regression and SVM: polluted day? |
| 3 | [`Week_03_Clustering_EM_PCA.ipynb`](Week_03_Clustering_EM_PCA.ipynb) | Mixture models and EM | 1. K-means, mixtures of Gaussians, hierarchical clustering → weather regimes · 2. PCA (weather axes, daily temperature curves) |
| 4 | [`Week_04_Hidden_Markov_Models.ipynb`](Week_04_Hidden_Markov_Models.ipynb) | Hidden Markov models | 1. HMM on daily air-quality classes and on daily / hourly ln PM2.5: hidden pollution regimes and forecasts |
| 5 | [`Week_05_Combining_Models_Ensembles.ipynb`](Week_05_Combining_Models_Ensembles.ipynb) | Combining models | 1. CART for polluted-day categorisation · 2. Ensembles (bagging, forests, boosting, voting, stacking) |

## Highlights (implemented from scratch with NumPy, then validated against scikit-learn / brute force)
* Week 1 – Bayes' rule on a medical test and on pollution data, PM2.5 is log-normal, over-fitting on a synthetic curve and on the annual temperature cycle; reusable `summarize()` EDA function.
* Week 2 – ML / ridge / robust / Bayesian linear regression (R² ≈ 0.64 on the held-out year from weather alone), **Bayesian logistic regression (Laplace approximation)**, kernel trick, SVM tuning, kernelised logistic regression.
* Week 3 – **K-means (k-means++)**, **EM for Gaussian mixtures**, hierarchical clustering → four interpretable weather regimes (BIC picks k = 4; the cold, calm regime has 76 % polluted days, the cold, dry, windy one 23 %), **PCA (eigen + SVD)**, factor analysis.
* Week 4 – **HMM** with scaled forward–backward, **Baum–Welch**, **Viterbi**, h-step prediction, **Markov-switching AR**; hidden regimes validated against wind (never used in fitting); **Kalman filter / RTS smoother** with missing data and weather inputs.
* Week 5 – Bayesian model averaging, bagging, **AdaBoost**, **gradient boosting**, GAM, **CART (Gini/entropy, pruning)**, bagging/OOB, voting, stacking, consensus clustering.

The notebooks report results as they came out, and where a first attempt was weak the models were improved rather than the evaluation relaxed (always chronological split, hyper-parameters chosen on 2010–2013 only):
* **Ensembles vs logistic regression (Week 5)** – tuning + context features lift every model (2014 accuracy: logistic regression 0.77 → 0.83, boosting 0.73 → 0.79); stacking now ties the tuned logistic regression, which still is not beaten even with 33,000 training hours — a property of this nearly additive problem, shown with a learning curve.
* **HMM forecasting (Week 4)** – a plain HMM loses to AR(1) for the next hours, so a *Markov-switching AR* (HMM with autoregressive emissions) is added: ≈ 7 % lower MAE/CRPS than AR(1) at 1 h, equal beyond ~12 h; best daily predictive density, small log-loss gain (accuracy within noise) for tomorrow's air-quality class.
* **Kalman gap-filling (Week 4)** – a latent AR(1) is no better than interpolation, but with the (always available) weather as known inputs the error for 24–48 h gaps falls by 20–28 %.

A few theory demos deliberately use small **synthetic** data where the truth is known (bagging on a sine curve, K-means vs. GMM on elongated clusters, PPCA dimension selection, the kernel-trick circles, HMM brute-force validation).

## Website (Vercel)
[`website/`](website/) is a one-page static site that summarises the results with the key figures (plain HTML + CSS, no build step).
To deploy: import this repo in Vercel and set **Root Directory** to `website` (Framework Preset: *Other*, leave build/output empty).

## Run it
```bash
pip install -r requirements.txt
jupyter lab        # open any Week_XX notebook and Run All (the CSV is already in data/)
```
