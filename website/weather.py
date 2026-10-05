"""Plot style shared by the website and the notebooks. The programs themselves read the data with pandas."""
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler


def use_style():
    """Clean, readable plots and tables."""
    pd.set_option("display.width", 110)
    pd.set_option("display.max_columns", 20)
    plt.rcParams.update({
        "figure.figsize": (9, 3.6), "figure.constrained_layout.use": True,
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.alpha": .25, "grid.linewidth": .8,
        "legend.frameon": False, "lines.linewidth": 2,
        "axes.prop_cycle": cycler(color=["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#72b7b2", "#eeca3b", "#9d755d"]),
    })
