"""Plot per-episode Reacher ablation results and paired bootstrap intervals."""

import argparse
import json
from pathlib import Path

import numpy as np

LABELS = {
    "baseline": "Normal inputs",
    "wrong_target": "Wrong target",
    "frozen": "Frozen image",
    "unrelated": "Unrelated images",
    "black": "Black image",
    "corrupted_proprio": "Corrupted proprioception",
}


def plot_results(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from evaluate_ablations import summarize

    directory = Path(directory)
    rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines() if line.strip()]
    summary = summarize(rows)
    if not summary:
        raise ValueError("No episode results to plot.")
    baseline = {row["seed"]: row["return"] for row in rows if row["condition"] == "baseline"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    rng = np.random.default_rng(42)
    positions = np.arange(len(summary))
    for i, entry in enumerate(summary):
        selected = [row for row in rows if row["condition"] == entry["condition"]]
        values = np.asarray([row["return"] for row in selected])
        delta = values - np.asarray([baseline[row["seed"]] for row in selected])
        jitter = rng.uniform(-0.15, 0.15, len(values))
        for axis, points, mean_key, ci_key in (
            (axes[0], values, "mean_return", "return_ci95"),
            (axes[1], delta, "mean_paired_delta", "paired_delta_ci95"),
        ):
            mean, interval = entry[mean_key], entry[ci_key]
            axis.scatter(i + jitter, points, alpha=0.22, s=12, color="tab:blue", zorder=2)
            axis.bar(i, mean, width=0.6, alpha=0.25, color="tab:blue", zorder=1)
            axis.vlines(i, interval[0], interval[1], color="black", linewidth=1.8, zorder=3)
            axis.scatter([i], [mean], color="black", s=22, zorder=4)
    labels = [LABELS[entry["condition"]] for entry in summary]
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_title("Evaluation return")
    axes[0].set_ylabel("Episode return")
    axes[1].set_title("Paired change from normal inputs")
    axes[1].set_ylabel("Return difference (condition − baseline)")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    fig.suptitle("Frozen image + proprioception policy: observation ablations")
    fig.supxlabel("Dots: episodes; black: mean and 95% episode-bootstrap interval (one trained checkpoint)")
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"ablation_comparison.{extension}", dpi=200)
    plt.close(fig)

    examples = [
        next((row for row in rows if row["condition"] == entry["condition"] and "trajectory" in row), None)
        for entry in summary
    ]
    if not all(examples):
        return
    fig, axes = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True, layout="constrained")
    for axis, row in zip(axes.flat, examples):
        with np.load(directory / row["trajectory"]) as trajectory:
            endpoint, target, fake = trajectory["endpoint"], trajectory["real_target"], trajectory["fake_target"]
            axis.plot(endpoint[:, 0], endpoint[:, 1], linewidth=1, label="Endpoint path")
            axis.scatter(*endpoint[0], marker="s", s=25, color="tab:blue", label="Start")
            axis.scatter(*target, color="green", marker="o", s=50, label="Real target")
            if np.isfinite(fake).all():
                axis.scatter(*fake, color="red", marker="x", s=60, label="Visible fake target")
        axis.set_title(f"{LABELS[row['condition']]} (return {row['return']:.1f})")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    fig.suptitle(f"Example endpoint trajectories: matched episode seed {examples[0]['seed']}")
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"example_trajectories.{extension}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Evaluation output directory")
    args = parser.parse_args()
    plot_results(args.directory)
    print(f"Saved ablation plots in {args.directory}")


if __name__ == "__main__":
    main()
