"""Compare original and balanced combined-model ablations side by side."""

import argparse
import json
from pathlib import Path

import numpy as np

from plot_ablations import LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("plots/five_seed_ablations"))
    parser.add_argument("--balanced", type=Path, default=Path("plots/balanced_five_seed_ablations"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/balanced_five_seed_ablations"))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    sources = {}
    for label, directory, offset, color in (
        ("Original combined", args.original, -.19, "tab:blue"),
        ("Balanced combined", args.balanced, .19, "tab:orange"),
    ):
        summary = {row["condition"]: row for row in json.loads((directory / "summary.json").read_text())}
        points = json.loads((directory / "per_seed_means.json").read_text())
        sources[label] = str(directory.resolve())
        if set(summary) != set(conditions):
            raise ValueError(f"Missing or unexpected ablation conditions in {directory}")
        x = np.arange(len(conditions)) + offset
        for axis, mean_key, sd_key in (
            (axes[0], "mean_return", "std_across_training_seeds"),
            (axes[1], "mean_paired_delta", "paired_delta_std_across_training_seeds"),
        ):
            mean = [summary[c][mean_key] for c in conditions]
            sd = [summary[c][sd_key] for c in conditions]
            axis.bar(x, mean, width=.35, color=color, alpha=.65, label=label,
                     yerr=sd, capsize=4, error_kw={"linewidth": 1.3}, zorder=2)
            for i, condition in enumerate(conditions):
                selected = sorted((row for row in points if row["condition"] == condition),
                                  key=lambda row: row["training_seed"])
                if len(selected) != summary[condition]["training_seeds"]:
                    raise ValueError(f"Inconsistent per-seed results in {directory}")
                axis.scatter(x[i] + np.linspace(-.09, .09, len(selected)),
                             [row[mean_key] for row in selected], s=17, color=color,
                             edgecolors="white", linewidths=.4, zorder=4)
    for axis in axes:
        axis.set_xticks(np.arange(len(conditions)), list(LABELS.values()), rotation=30, ha="right")
        axis.grid(axis="y", alpha=.25)
        axis.axhline(0, color="gray", linewidth=.8)
        axis.legend(frameon=False)
    axes[0].set_title("Evaluation return")
    axes[0].set_ylabel("Mean episode return")
    axes[1].set_title("Paired change from each model's normal inputs")
    axes[1].set_ylabel("Return difference (condition − baseline)")
    fig.suptitle("Original versus balanced combined model: observation ablations")
    fig.supxlabel("5 training seeds per model · 100 matched episodes per condition per seed\n"
                  "Bars: mean; error bars: ±1 sample SD across trained models; dots: individual model means")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = args.output_dir / f"original_vs_balanced_ablation_comparison.{extension}"
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    plt.close(fig)
    (args.output_dir / "original_vs_balanced_metadata.json").write_text(json.dumps({
        "sources": sources, "error_bars": "Sample SD across per-training-seed episode means, ddof=1",
        "paired_change": "Within-model, within-episode condition minus baseline; not an inter-model paired test",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
