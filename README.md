# Machine Learning Practice (21CSC305P)

Every question of the **21CSC305P – Machine Learning** syllabus (theory topics and practice tasks) as short, self-contained Python on one small dataset.
The same code is used in three places, so they can never disagree:

* [`website/code/week1.py` … `week5.py`](website/code/): the single source of truth, one file per unit, one block per question.
* **The website** ([`website/`](website/)): every question is shown with its code and a **▶ Run** button. Python runs in the browser ([Pyodide](https://pyodide.org)), so the output and plots are produced live. Nothing is pre-generated. Light and dark mode.
* **The notebooks** (`Week_0X_*.ipynb`): built from the same files by [`build_notebooks.py`](build_notebooks.py) and saved with their outputs and SVG plots.

## The dataset: daily weather in Seattle, 2012–2015
One row per day, 1,461 days, no missing values: precipitation (mm), highest and lowest temperature (°C), wind speed, and a weather label (sun, rain, drizzle, fog, snow).

* Source: NOAA National Climatic Data Center, as packaged in [vega-datasets](https://github.com/vega/vega-datasets) (`seattle-weather.csv`). U.S. Government work, public domain.
* [`website/data/seattle_weather.csv`](website/data/seattle_weather.csv) is the data and [`website/weather.py`](website/weather.py) loads it and adds a few columns (`rain`, `rain_tomorrow`, `temp_tomorrow`, month as a circle).
* The `weather` label was built by the dataset's authors from the other columns, so it is never used as an input, only to check clusters in week 3.
* Models are trained on **2012–2014** and tested on **2015**; results are compared with a simple baseline ("tomorrow = today").

## Syllabus checklist (every topic → the question that covers it)

| Unit | Syllabus topic | Question (in `website/code/weekN.py`) |
|---|---|---|
| 1 | Machine learning: what and why; supervised and unsupervised learning | Machine learning: what and why? Supervised and unsupervised learning |
| 1 | Polynomial curve fitting | Polynomial curve fitting |
| 1 | Discrete random variables, fundamental rules, Bayes' rule | Discrete random variables, fundamental rules and Bayes' rule |
| 1 | Independence and conditional independence | Independence and conditional independence |
| 1 | Continuous random variables, quantiles, mean and variance, probability densities | Continuous random variables: probability density, quantiles, mean and variance |
| 1 | Expectation and covariance | Expectation and covariance |
| 1 | **Practice 1** – import, load and view a dataset | Devise a program to import, load and view the dataset |
| 1 | **Practice 2** – summary and statistics | Create a program to display the summary and statistics of the dataset |
| 2 | Maximum likelihood estimation: least squares | **Practice 1** – Implement linear regression to perform prediction (maximum likelihood = least squares) |
| 2 | Robust linear regression | Robust linear regression |
| 2 | Ridge regression | Ridge regression |
| 2 | Bayesian linear regression | Bayesian linear regression |
| 2 | Discriminant function, probabilistic generative and discriminative models | Discriminant functions, generative and discriminative models for classification |
| 2 | Laplace approximation, Bayesian logistic regression | **Practice 2** – Implement Bayesian logistic regression (Laplace approximation) for classification |
| 2 | Kernel functions, kernels in GLMs, kernel trick, SVMs | **Practice 2** – Implement SVM for classification: kernel functions, kernels in GLMs and the kernel trick |
| 3 | Measuring dissimilarity | Clustering: measuring dissimilarity |
| 3 | K-means clustering | **Practice 1** – Implement K-means clustering to categorize data (also evaluates the output: elbow, silhouette) |
| 3 | Mixtures of Gaussians, an alternative view of EM | **Practice 1** – Implement mixtures of Gaussians (EM) to categorize data: an alternative view of EM |
| 3 | Hierarchical clustering, evaluating the output of clustering | **Practice 1** – Implement hierarchical clustering to categorize data (and evaluate the clusterings) |
| 3 | PCA | **Practice 2** – Create a program to perform PCA |
| 3 | Choosing the number of latent dimensions | Choosing the number of latent dimensions |
| 3 | Factor analysis | Factor analysis |
| 4 | Sequential data, Markov models | Sequential data: Markov models |
| 4 | Forward and backward algorithms, sum-product algorithm, scaling factors | The forward and backward algorithms, the sum-product algorithm and scaling factors |
| 4 | Viterbi algorithm | The Viterbi algorithm |
| 4 | HMM, maximum likelihood for the HMM (Baum–Welch) | **Practice 1** – Implement HMM to predict the sequential data |
| 4 | Linear dynamical systems | Linear dynamical systems (the Kalman filter) |
| 5 | Bayesian model averaging | Bayesian model averaging |
| 5 | Boosting, adaptive basis function models | Boosting and adaptive basis function models |
| 5 | Generalized additive models | Generalized additive models (GAM) |
| 5 | CART | **Practice 1** – Implement CART learning algorithm to perform categorization |
| 5 | Ensemble learning | **Practice 2** – Implement ensemble learning models to perform classification |

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

## Adding or changing a question
Edit the block in `website/code/weekN.py`. A block starts with `# %% Tag | Title`, then `#:` lines (the explanation shown above the code), then the code. Keep each question self-contained (its own imports) because every **Run** starts from a clean namespace. Then run `python build_notebooks.py` to refresh the notebooks.
