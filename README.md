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
* **Unit 3 clustering:** direct K-means, Gaussian mixture and Ward calls fit all 13 standardized Wine measurements. Each uses three groups as an explicit teaching choice. The four graphs are K-means groups/centres, GMM assignment confidence, the actual Ward dendrogram, and a K-means elbow curve. Silhouette scores use all 13 features. Cultivar labels are not used for clustering. Cluster numbers are arbitrary, and the PCA drawing keeps only part of the variance.
* **Unit 3 PCA:** four simple figures show all 13 scree bars, cumulative variance with a 95% reference, PC1–PC2 scores, and horizontal feature-weight bars for PC1/PC2. Loadings here are component coefficients, not correlations. The scatter axes report their variance percentages, so a two-dimensional view is not presented as retaining all the information. Component signs are arbitrary.
* **Unit 4:** count probabilities on 2012–2014 and evaluate 2015. The two-state Viterbi function uses log probabilities. Decoding uses the whole observation sequence; rolling next-day forecasts use only yesterday's known state. Report majority and persistence baselines. Two graphs show actual/decoded/forecast timelines and a five-day wet probability curve. Temperature range is a weak observation, so errors are expected.
* **Unit 5:** one stratified 80/20 split and the same five training folds. CART has a depth-3 tree drawing, basic training/CV depth curves and a test confusion matrix. Ensembles use direct model calls and hard majority voting, with a simple accuracy bar chart and confusion matrices. Report test accuracy, macro F1 and a majority baseline. Use training folds for comparison; do not tune models or select a winner on test results. Ensembles need not beat a single tree.

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

ML checks: `python -m unittest discover -s tests -p 'test_ml_examples.py'`. These check the real datasets, ordinary Python loops, consistent Ward assignments, PCA weights and reconstruction, HMM probabilities and long-sequence decoding, and held-out/CV split isolation. They do not require artificially high scores or disagreement between clustering algorithms.

## Change a question

Edit its block in `website/code/weekN.py`. A block starts with `# %% Tag | Title`, followed by `#:` explanation lines, then code. Keep each question self-contained because every **Run** starts from a clean namespace. Run `python build_notebooks.py` to refresh saved notebooks and outputs.
