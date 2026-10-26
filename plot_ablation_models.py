# Compare the ablation results of several trained models in one bar plot.
# Usage: python plot_ablation_models.py dirA dirB --labels "Model A" "Model B" --output plots/name

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_ablations import LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path, help="Ablation output folders with summary.json")
    parser.add_argument("--labels", nargs="+", required=True, help="One legend label per folder")
    parser.add_argument("--output", type=Path, default=Path("plots/ablation_models"))
    parser.add_argument("--title", default="Observation ablations")
    args = parser.parse_args()
    if len(args.labels) != len(args.directories):
        parser.error("--labels needs one label per folder")

    # read the summary of every model, keyed by condition
    summaries = []
    for directory in args.directories:
        rows = json.loads((directory / "summary.json").read_text())
        summaries.append({row["condition"]: row for row in rows})
    conditions = [c for c in LABELS if all(c in summary for summary in summaries)]

    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    x = np.arange(len(conditions))
    width = 0.8 / len(summaries)
    for i, (summary, label) in enumerate(zip(summaries, args.labels)):
        means = np.array([summary[c]["mean_return"] for c in conditions])
        intervals = np.array([summary[c]["return_ci95"] for c in conditions])
        # error bars go from the lower to the upper end of the 95% bootstrap interval
        errors = np.stack([means - intervals[:, 0], intervals[:, 1] - means])
        offset = (i - (len(summaries) - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=errors, capsize=3, label=label)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=6)

    ax.set_xticks(x, [LABELS[c] for c in conditions], rotation=20, ha="right")
    ax.set_ylabel("Mean episode return (100 episodes)")
    ax.set_ylim(0, 1100)
    ax.set_title(args.title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.supxlabel("Error bars: 95% episode-bootstrap interval for one trained checkpoint", fontsize=9)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output.with_suffix("." + extension), dpi=200)
    print("Saved", args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
