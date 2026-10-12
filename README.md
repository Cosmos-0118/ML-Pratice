# Machine Learning Practice (21CSC305P)

Nine practice questions over five units, using real datasets suited to each algorithm.

* [`website/code/week1.py` … `week5.py`](website/code/): the source for every practice question.
* **The website** ([`website/`](website/)): pick a unit, read the question, press **Run**. Python runs in the browser ([Pyodide](https://pyodide.org)); plots are shown one at a time with the arrow keys. Light and dark mode.
* **The notebooks** (`Week_0X_*.ipynb`): built from those same source files by [`build_notebooks.py`](build_notebooks.py) and saved with executed outputs.

## Real datasets

| Units | Dataset | Purpose and source |
|---|---|---|
| 1, 2, 4 | Seattle daily weather, 2012–2015; 1,461 consecutive days | Observed precipitation, temperatures, wind and weather categories. [Vega datasets source](https://github.com/vega/vega-datasets/blob/main/data/seattle-weather.csv); bundled as [`website/data/seattle_weather.csv`](website/data/seattle_weather.csv). |
| 3 | UCI Wine; 178 samples, 13 chemical measurements | Multivariate clustering and dimensionality reduction. [UCI source and attribution](https://archive.ics.uci.edu/dataset/109/wine), CC BY 4.0; loaded through [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html). |
| 5 | Wisconsin Diagnostic Breast Cancer; 569 samples, 30 measurements | Binary classification of malignant/benign samples. [UCI source and attribution](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), CC BY 4.0; loaded through [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html). |

No rows or labels are generated. Wine and breast cancer data are bundled in scikit-learn, so no separate data download is needed once the browser has loaded the Python packages. The examples are educational; the classifier is not a clinical diagnostic tool.

## What the examples demonstrate

* **Unit 2:** next-day temperature and precipitation prediction. The first 80% of days train the models; the final 20% evaluate them. Scaling is fitted only on training inputs. Regression includes a persistence baseline.
* **Unit 3 clustering:** all models fit the 13 standardized Wine measurements. K-means selects its cluster count by silhouette; GMM selects component count and covariance structure by BIC. Ward clustering uses the K-means count to provide a common comparison, and the assignments come from the exact dendrogram tree shown. Cultivar labels are used only for post-fit adjusted Rand agreement, never for model selection. Cluster numbers are arbitrary and are not aligned between algorithms.
* **Clustering figures:** one shared PCA comparison with K-means centroids and correctly projected GMM covariance ellipses; K-means elbow/separation curves and silhouette profiles; GMM BIC and soft assignment confidence; Ward dendrogram cut and standardized feature profiles. Scores use all 13 features; the two-dimensional projection can hide structure. Silhouette and BIC can favour different counts because they measure different properties.
* **Unit 3 PCA:** six figures cover input correlations, all 13 scree bars, cumulative variance with 90%/95% thresholds, PC1–PC2/PC1–PC3/PC2–PC3 score panels, all-feature correlation loadings, and a symmetric SVD biplot. Loadings are feature-to-score correlations. The biplot uses `U sqrt(S)` for samples and `V sqrt(S)` for features, so their product reconstructs the two-PC approximation; arrows are not arbitrarily enlarged. Six well-represented feature arrows keep it readable; the loadings heatmap and printed table cover all 13. Signs of PCA components are arbitrary.
* **Unit 4:** supervised HMM estimation on 2012–2014 and evaluation on 2015, with smoothed probability tables and Viterbi in log space. Decoding uses the whole test observation sequence and is reported separately from rolling next-day forecasts, which use only yesterday's known state. Forecasts have persistence/majority baselines and a Brier score. Temperature range is a weak observation, so this simple real-weather HMM can perform poorly. Multi-day probability curves demonstrate transition dynamics without claiming future observations are known.
* **Unit 5:** one fixed stratified 80/20 split and the same five training folds for comparison. A depth-3 CART tree has a readable tree diagram, training/CV depth curves and a held-out confusion matrix. Ensembles have CV mean/standard deviation, held-out accuracy and macro F1, a majority baseline, and confusion matrices. Logistic scaling in soft voting happens inside the CV pipeline. Held-out results illustrate generalization; do not tune models or select a winner on those test results. Ensembles need not beat a single tree.

## Practice questions

| Unit | Practice question |
|---|---|
| 1 – Introduction | 1. Devise a program to import, load and view dataset |
| | 2. Create a program to display the summary and statistics of the dataset |
| 2 – Linear models for regression | 1. Implement linear regression to perform prediction |
| | 2. Implement Bayesian logistic regression and SVM for classification |
| 3 – Mixture models and EM | 1. Implement K-means clustering, mixtures of Gaussians and Hierarchical clustering algorithm to categorize data |
| | 2. Create a program to perform PCA |
| 4 – Hidden Markov models | 1. Implement HMM to predict the sequential data |
| 5 – Combining models | 1. Implement CART learning algorithms to perform categorization |
| | 2. Implement Ensemble learning models to perform classification |

## Run it

Website (any static server, no build step):

```bash
cd website && python3 -m http.server 8000
# Open http://localhost:8000
```

Notebooks:

```bash
pip install -r requirements.txt
jupyter lab
python build_notebooks.py
```

To deploy on Vercel: import this repo and set **Root Directory** to `website` (Framework Preset: *Other*, leave build/output empty).

Website regression tests (Node.js 22.13+ or 24+): `npm ci && npm test`. These cover output formatting, browser dataset loading and Run button behaviour; Node.js is needed only for tests.

ML checks: `python -m unittest discover -s tests -p 'test_ml_examples.py'`. These check the real dataset, chronology, projected GMM covariances, consistent Ward assignments, exact PCA loadings and biplot reconstruction, HMM probabilities and long-sequence decoding, and held-out/CV split isolation. They do not require artificially high scores or disagreement between clustering algorithms.

## Change a question

Edit its block in `website/code/weekN.py`. A block starts with `# %% Tag | Title`, followed by `#:` explanation lines, then code. Keep each question self-contained because every **Run** starts from a clean namespace. Run `python build_notebooks.py` to refresh saved notebooks and outputs.
