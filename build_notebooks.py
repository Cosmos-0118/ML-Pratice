"""Builds the five Week_XX notebooks (one per unit, practice questions only) from website/code/ and runs them.

The website shows the same code (website/code/weekN.py), so the notebooks and the site can never disagree.
Usage:  pip install nbformat nbclient ipykernel   then   python build_notebooks.py
"""
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).parent
WEEKS = {
    1: ("Week_01_Introduction_to_ML", "Introduction"),
    2: ("Week_02_Linear_Models", "Linear models for regression"),
    3: ("Week_03_Clustering_EM_PCA", "Mixture models and EM"),
    4: ("Week_04_Hidden_Markov_Models", "Hidden Markov models"),
    5: ("Week_05_Combining_Models_Ensembles", "Combining models"),
}
SETUP = '''%config InlineBackend.figure_format = "svg"
import sys
sys.path.insert(0, "website")          # weather.py and the data live in website/
from weather import use_style
use_style()'''


def parse(path):
    """Returns a list of (tag, title, description, code)."""
    _, *blocks = re.split(r"(?m)^# %% ", path.read_text())
    questions = []
    for block in blocks:
        first, _, body = block.partition("\n")
        tag, title = (s.strip() for s in first.split("|", 1))
        lines = body.splitlines()
        description = " ".join(l[2:].strip() for l in lines if l.startswith("#:"))
        code = "\n".join(l for l in lines if not l.startswith("#:")).strip()
        questions.append((tag, title, description, code))
    return questions


for number, (name, title) in WEEKS.items():
    questions = parse(ROOT / "website" / "code" / f"week{number}.py")
    cells = [nbformat.v4.new_markdown_cell(
        f"# Unit {number} — {title}\n\nThe practice questions of this unit (21CSC305P). "
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
