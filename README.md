# Machine Learning Practice (21CSC305P)

One Jupyter notebook per week of the **21CSC305P – Machine Learning** syllabus, all on one small dataset.
Each notebook is short, plain Python, and is saved **with its outputs and graphs**, so it can be read directly on GitHub.
It also re-runs top to bottom.

## The dataset: daily weather in Seattle, 2012–2015
One row per day, 1,461 days, no missing values: precipitation (mm), highest and lowest temperature (°C), wind speed, and a weather label (sun, rain, drizzle, fog, snow).

* Source: NOAA National Climatic Data Center, as packaged in [vega-datasets](https://github.com/vega/vega-datasets) (`seattle-weather.csv`). U.S. Government work, public domain.
* The CSV is in [`data/seattle_weather.csv`](data/seattle_weather.csv). [`weather.py`](weather.py) loads it and adds a few columns (`rain`, `rain_tomorrow`, `temp_tomorrow`, month as a circle).
* The `weather` label was built by the dataset's authors from the other columns, so it is **never used as an input**, only to check clusters in week 3.
* Evaluation never looks at the future: models are trained on **2012–2014** and tested on **2015**, and every result is compared with a simple baseline ("tomorrow = today").

| Week | Notebook | Unit | Practice tasks covered |
|---|---|---|---|
| 1 | [`Week_01_Introduction_to_ML.ipynb`](Week_01_Introduction_to_ML.ipynb) | Introduction | Load and view the data · summary and statistics (missing values, skew, seasons, correlation), Bayes' rule, over-fitting |
| 2 | [`Week_02_Linear_Models.ipynb`](Week_02_Linear_Models.ipynb) | Linear models for regression and classification | Linear regression: tomorrow's temperature · Bayesian logistic regression and SVM: will it rain tomorrow? |
| 3 | [`Week_03_Clustering_EM_PCA.ipynb`](Week_03_Clustering_EM_PCA.ipynb) | Mixture models and EM | K-means, Gaussian mixtures (EM), hierarchical clustering → weather types · PCA |
| 4 | [`Week_04_Hidden_Markov_Models.ipynb`](Week_04_Hidden_Markov_Models.ipynb) | Hidden Markov models | HMM on the daily rain sequence: hidden wet and dry spells, forecasting |
| 5 | [`Week_05_Combining_Models_Ensembles.ipynb`](Week_05_Combining_Models_Ensembles.ipynb) | Combining models | CART · bagging, random forest, boosting, voting, stacking |

## What is written from scratch (NumPy), then checked against scikit-learn or brute force
* Week 2 – least squares (normal equation), Bayesian linear regression, Bayesian logistic regression (Laplace approximation).
* Week 3 – K-means and EM for a Gaussian mixture.
* Week 4 – HMM: forward algorithm, Baum–Welch, Viterbi (forward checked against brute force).
* Week 5 – one decision-tree split (Gini) and bagging.

## Results, as they came out
* Rain tomorrow: logistic regression 73.1% against 70.3% for "tomorrow = today". Trees, boosting, stacking and SVMs do not clearly beat it.
* The HMM finds clear wet and dry spells (about eight days each, matching wind and temperature it never saw) but does not predict tomorrow better than "tomorrow = today".
* Clustering gives readable weather types, but there is no sharp number of clusters.
* The 2015 test year has 364 days, so one accuracy is uncertain by about ±2.4 points.

## Website
[`website/`](website/) is a small static site: a home page, one page per week with the key figures and an **editable, runnable code cell** (Python in the browser via [Pyodide](https://pyodide.org); the code is in `website/code/`), a "what did not work" page, light and dark mode, and each notebook exported as HTML (`website/notebooks/`). No build step and no server.
To deploy on Vercel: import this repo and set **Root Directory** to `website` (Framework Preset: *Other*, leave build/output empty).
`website/weather.py` and `website/data/` are copies of the files in the repo root, so keep them in sync if you change them.

## Run it
```bash
pip install -r requirements.txt
jupyter lab        # open any Week_XX notebook and Run All (the CSV is already in data/)
```
