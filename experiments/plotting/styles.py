"""Matplotlib defaults for conference-style diagrams."""

from typing import Any, Dict

import matplotlib


# Use a non-interactive backend so scripts run headless by default.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PAPER_RC: Dict[str, Any] = {
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_paper_style() -> None:
    """Apply the default paper style to matplotlib."""
    plt.rcParams.update(PAPER_RC)


# Default colors and line styles for the slice plots.
COLORS = {
    "do": "#d62728",  # red
    "obs": "#1f77b4",  # blue
}

LINESTYLES = {
    "circuit": "-",
    "empirical": "--",
}
