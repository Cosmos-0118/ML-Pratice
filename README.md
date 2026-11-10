# Machine Learning Practice (21CSC305P)

Nine practice questions over five units, using real datasets suited to each algorithm.

* [`website/code/week1.py` … `week5.py`](website/code/): the source for every practice question.
* **The website** ([`website/`](website/)): pick a unit, read the question, press **Run**. Python runs in the browser ([Pyodide](https://pyodide.org)); plots are shown one at a time with the arrow keys. Light and dark mode.
* **The notebooks** (`Week_0X_*.ipynb`): built from those same source files by [`build_notebooks.py`](build_notebooks.py) and saved with executed outputs.

## Real datasets

| Units | Dataset | Purpose and source |
|---|---|---|
| 1, 2 | Seattle daily weather, 2012–2015; 1,461 consecutive days | Observed precipitation, temperatures, wind and weather categories. [Vega datasets source](https://github.com/vega/vega-datasets/blob/main/data/seattle-weather.csv); bundled as [`website/data/seattle_weather.csv`](website/data/seattle_weather.csv). |
| 4 | Sleep-EDF Expanded; eight people, 21,948 nonoverlapping 30-second signal segments | 15 EEG/EOG log-bandpower features; 21,947 expert-labelled stages. [PhysioNet source, credit and license](https://physionet.org/content/sleep-edfx/1.0.0/); [bundled feature CSV](website/data/sleep_edf.csv) and [source checksums/preparation](website/data/sleep_edf_provenance.json). |
| 3, 5 | UCI Wine; 178 samples, 13 chemical measurements | Unit 3: clustering and dimensionality reduction. Unit 5: predicting the cultivar with trees and ensembles. [UCI source and attribution](https://archive.ics.uci.edu/dataset/109/wine), CC BY 4.0; loaded through [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html). |

The practice programs use only real observations and labels. (The Forest Lab's toy shapes in Unit 5 are generated on purpose and labelled as toys; its Wine mode is real data.) Unit 4 derives features from original recorded signals; its bundled CSV is a prepared feature extract, not raw EEG. Sleep-EDF credit: Bob Kemp and colleagues / PhysioNet; Open Data Commons Attribution License v1.0 (required citations are on the linked source page). Wine data is bundled in scikit-learn, so no separate data download is needed once the browser has loaded the Python packages. The examples are educational; the classifier is not a clinical diagnostic tool.

## What the examples demonstrate

* **Unit 2:** next-day temperature and precipitation prediction. The first 80% of days train the models; the final 20% evaluate them. Scaling is fitted only on training inputs. Regression includes a persistence baseline.
* **Unit 3 clustering:** direct K-means, Gaussian mixture and Ward calls fit all 13 standardized Wine measurements. Each uses three groups as an explicit teaching choice. The four graphs are K-means groups/centres, GMM assignment confidence, the actual Ward dendrogram, and a K-means elbow curve. Silhouette scores use all 13 features. Cultivar labels are not used for clustering. Cluster numbers are arbitrary, and the PCA drawing keeps only part of the variance.
* **Unit 3 PCA:** four simple figures show all 13 scree bars, cumulative variance with a 95% reference, PC1–PC2 scores, and horizontal feature-weight bars for PC1/PC2. Loadings here are component coefficients, not correlations. The scatter axes report their variance percentages, so a two-dimensional view is not presented as retaining all the information. Component signs are arbitrary.
* **Unit 4:** an interactive explorer first: press **Run** on any of the eight people (or **Run all 8**) and Python, live in the browser, trains on the other seven and decodes that person. It shows the scores against both baselines, a zoomable night (expert, independent and HMM stages, mistakes, EEG, forward–backward confidence with a hover read-out), the confusion matrix, and what the model learned. The **Full program** tab keeps the original all-people code and five graphs. The explorer runs the same model code as `week4.py` (everything above its `# 2. Leave one person out` line) plus the JSON glue in `website/code/sleep_bridge.py`. Sleep-stage recognition with a supervised Gaussian HMM. Fit scaling, emissions, initial probabilities and transitions on seven people and decode the eighth person's complete recording; repeat for all eight. Original stages 3/4 are combined as N3. Movement/unscored gaps break training sequences; test labels never select decoding boundaries and are used only for scoring. Viterbi and forward–backward operate in log space. Compare independent Gaussian classification and the training-majority baseline, reporting pooled accuracy, macro F1, per-person scores and per-stage F1. Five figures show EEG inputs, expert/independent/decoded timelines, learned transitions, posterior probabilities and held-out confusion. The example view is fixed at hours 8–16 for SC400, independent of expert labels. The 86.13% accuracy / 0.696 macro F1 improvement over 85.72% / 0.670 independently is modest; Wake dominates full recordings and N1 F1 is only about 0.246. These eight people are an educational subset; decoding is retrospective, probabilities are uncalibrated, and this is not a clinically validated model.
* **Unit 5:** one stratified 80/20 split and the same five training folds. CART has a depth-3 tree drawing, basic training/CV depth curves and a test confusion matrix. Ensembles use direct model calls and hard majority voting, with a simple accuracy bar chart and confusion matrices. Report test accuracy, macro F1 and a majority baseline. Use training folds for comparison; do not tune models or select a winner on test results. Ensembles need not beat a single tree.

## Forest Lab (Unit 5, interactive)

[`website/forest-lab/`](website/forest-lab/) is the interactive side of Unit 5, embedded above both practice questions and also open as its own page (`/forest-lab/`). Pick the real **Wine** data (any two of the 13 features) or a toy shape, then watch **CART**, a **random forest**, **AdaBoost** or a **voting ensemble** (tree + kNN + Naive Bayes) carve up a 3D map: height is how sure the model is, valleys are decision boundaries. The CART tree is drawn split by split, the slider grows an ensemble one learner at a time, hovering shows how every learner votes, and a table compares all models. Toy shapes also let you click to draw your own points. The models are plain JavaScript ([`ml.mjs`](website/forest-lab/ml.mjs), tested in `tests/forest_lab.test.cjs`); [three.js](https://threejs.org) (MIT, vendored in `vendor/`) draws the scene. Held-out points are for understanding over-fitting only, never for tuning.

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

ML checks: `python -m unittest discover -s tests -p 'test_ml_examples.py'`. These check the real datasets, ordinary Python loops, consistent Ward assignments, PCA weights and reconstruction, Sleep-EDF provenance, held-out person isolation, Gaussian emissions, HMM paths/posteriors against exhaustive enumeration and long-sequence decoding, and held-out/CV split isolation. They do not require artificially high scores or disagreement between clustering algorithms.

## Change a question

Edit its block in `website/code/weekN.py`. A block starts with `# %% Tag | Title`, followed by `#:` explanation lines, then code. Keep each question self-contained because every **Run** starts from a clean namespace. Run `python build_notebooks.py` to refresh saved notebooks and outputs, or `python build_notebooks.py --unit 4` to rebuild only Unit 4.

## Rebuild the sleep features

The website runs directly from the bundled feature CSV; it needs no EDF reader, signal-processing download or database server. The dataset preview shows 200 of the 21,948 rows and offers the full CSV plus preparation/source checksums.

To reproduce the CSV from the original EDFs, download the first-night PSG and Hypnogram files for SC400–SC407 from PhysioNet into one folder, install `pyedflib` alongside the normal requirements, then run `python prepare_sleep_data.py --source /path/to/recordings`. The exporter checks all 16 files against the pinned source hashes before extracting features. It retains recording identifiers, original order and the single unscored epoch. The 15 features use five fixed frequency bands in each of two EEG channels and horizontal EOG, without label-dependent preprocessing. Full extraction details and hashes are in `website/data/sleep_edf_provenance.json`. Raw EDFs and local experiment outputs stay outside Git.
