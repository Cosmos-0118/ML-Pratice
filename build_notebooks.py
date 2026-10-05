"""Builds the five Week_XX notebooks from the question files in website/code/ and runs them.

The website shows the same code (website/code/weekN.py), so the notebooks and the site can never disagree.
Usage:  pip install nbformat nbclient ipykernel   then   python build_notebooks.py
"""
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).parent
WEEKS = {
    1: ("Week_01_Introduction_to_ML", "Introduction to Machine Learning"),
    2: ("Week_02_Linear_Models", "Linear Models for Regression and Classification"),
    3: ("Week_03_Clustering_EM_PCA", "Mixture Models, EM, Clustering and PCA"),
    4: ("Week_04_Hidden_Markov_Models", "Hidden Markov Models"),
    5: ("Week_05_Combining_Models_Ensembles", "Combining Models: Boosting, CART and Ensembles"),
}
SETUP = '''%config InlineBackend.figure_format = "svg"
import sys
sys.path.insert(0, "website")          # weather.py and the data live in website/
from weather import use_style
use_style()'''


def parse(path):
    """Returns the syllabus line and a list of (tag, title, description, code)."""
    text = path.read_text()
    head, *blocks = re.split(r"(?m)^# %% ", text)
    syllabus = " ".join(l[2:].strip() for l in head.splitlines() if l.startswith("#:"))
    questions = []
    for block in blocks:
        first, _, body = block.partition("\n")
        tag, title = (s.strip() for s in first.split("|", 1))
        lines = body.splitlines()
        description = " ".join(l[2:].strip() for l in lines if l.startswith("#:"))
        code = "\n".join(l for l in lines if not l.startswith("#:")).strip()
        questions.append((tag, title, description, code))
    return syllabus, questions


for number, (name, title) in WEEKS.items():
    syllabus, questions = parse(ROOT / "website" / "code" / f"week{number}.py")
    cells = [nbformat.v4.new_markdown_cell(
        f"# Week {number} — {title}\n\n**Syllabus topics:** {syllabus}\n\n"
        "Every question below is self-contained: run any cell on its own. All questions use the Seattle daily weather data "
        "(`website/data/seattle_weather.csv`); models are trained on 2012–2014 and tested on 2015.")]
    cells.append(nbformat.v4.new_code_cell(SETUP))
    for tag, q_title, description, code in questions:
        cells.append(nbformat.v4.new_markdown_cell(f"## {tag} — {q_title}\n\n{description}"))
        cells.append(nbformat.v4.new_code_cell(code))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=600, resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, ROOT / f"{name}.ipynb")
    print("wrote", name, f"({len(questions)} questions)")
