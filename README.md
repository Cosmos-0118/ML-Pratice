# Machine Learning Practice (21CSC305P)

Every **practice question** of the 21CSC305P Machine Learning syllabus (9 questions over 5 units) as short Python, all on one small dataset.

* [`website/code/week1.py` … `week5.py`](website/code/): the single source of truth, one file per unit, one block per question.
* **The website** ([`website/`](website/)): pick a unit, read the question, press **Run**. Python runs in the browser ([Pyodide](https://pyodide.org)) and the output replaces the code in the same box; plots are shown one at a time (← → keys). Light and dark mode.
  Captured text is highlighted by [`website/output.js`](website/output.js), independently of the Python code. Labels, table headings, numbers, dates, strings, and data types use the editor's theme colours; the original text and alignment are preserved.
* **The notebooks** (`Week_0X_*.ipynb`, one per unit): built from the same files by [`build_notebooks.py`](build_notebooks.py) and saved with their outputs.

## The dataset: synthetic daily weather regimes, 2012–2015
One row per day, 1,461 consecutive days, no missing values: precipitation (mm), highest and lowest temperature (°C), wind speed (m/s), and a generating weather-regime label (sun, cloudy, rain, snow).

**This is synthetic teaching data, not observed weather or a realistic climate model.** Its four regimes deliberately have distinct measurements so clustering is visible. Regimes persist over consecutive days so HMMs and next-day prediction have a meaningful sequence. Classification is intentionally easy; high accuracy demonstrates the algorithms on this simulation and does not establish performance on real weather.

* The data was simulated using fixed seeds (42 for transitions, 142 for measurement noise). Each day stays in its regime with probability 0.88, or moves to each other regime with probability 0.04. Measurements vary around that regime's centre; dry regimes have zero precipitation, and wet regimes have positive precipitation. The simulation has no seasonal effects.
* [`website/data/weather_regimes.csv`](website/data/weather_regimes.csv) is the shared data. Every program reads it itself with `pd.read_csv("data/weather_regimes.csv")`.
* The code is written for beginners: one step per line, a comment above each step, and scikit-learn for the standard methods.
* Unit 2 trains on the first 80% of days and tests on the final 20%. Unit 3 uses only the four numeric features to cluster, evaluates silhouette and agreement with the known generating labels, and draws a shared PCA projection. Unit 4 estimates a supervised HMM from 2012–2014 and decodes 2015 using log probabilities. Unit 5 uses an 80/20 stratified split (`random_state=42`) for same-day classification, with every class represented in both sets.
* Cluster IDs have no intrinsic meaning and can differ between algorithms. PCA is a projection and may hide some separation; quality scores use all four standardized features. Ensembles are compared fairly and need not outperform a single tree on an easy dataset.

## The practice questions

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
cd website && python3 -m http.server 8000      # then open http://localhost:8000
```
Notebooks:
```bash
pip install -r requirements.txt
jupyter lab                                    # open any Week_0X notebook and Run All
python build_notebooks.py                      # rebuilds all five notebooks from website/code/ and re-runs them
```
To deploy on Vercel: import this repo and set **Root Directory** to `website` (Framework Preset: *Other*, leave build/output empty).

Website regression tests (Node.js 22.13+ or 24+): `npm ci && npm test`. These check output formatting and Run button behaviour; Node.js is only needed for tests.

Dataset and ML checks: `python -m unittest discover -s tests -p 'test_ml_examples.py'`. These check physical ranges and chronology, separation for all three clustering methods, HMM probability tables and long-sequence decoding, class coverage and macro F1, and the chronological forecasting split.

## Adding or changing a question
Edit the block in `website/code/weekN.py` (N = unit number). A block starts with `# %% Tag | Title`, then `#:` lines (the explanation shown above the code), then the code. Keep each question self-contained (its own imports) because every **Run** starts from a clean namespace. Then run `python build_notebooks.py` to refresh the notebooks.
