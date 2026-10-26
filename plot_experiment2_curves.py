# Plot the experiment 2 runs (images + full state) on top of the experiment 1 seed means.
# Experiment 1 comes from plots/seed_comparison/mean_std_data.csv (3 seeds, already smoothed),
# so our runs get the same 5-evaluation moving average to keep the curves comparable.

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from plot_learning_curves import load_curve
from plot_seed_comparison import smooth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", type=Path, default=Path("plots/seed_comparison/mean_std_data.csv"))
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "results/experiment2/reacher_hard_images_state_seed0",
            "results/experiment2/reacher_hard_images_state_dropout_seed0",
        ],
    )
    parser.add_argument("--labels", nargs="+", default=["Images + full state", "Images + full state + modality dropout"])
    parser.add_argument("--max-step", type=float, default=5.1e5, help="Cut every curve at this many environment steps")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/experiment2/plots/learning_curves_vs_experiment1"))
    args = parser.parse_args()
    if len(args.labels) != len(args.runs):
        parser.error("--labels needs one label per run")

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")

    # experiment 1: mean and standard deviation over seeds, only the environment step axis
    baseline = {}
    with args.baseline_csv.open() as stream:
        for row in csv.DictReader(stream):
            if row["axis"] != "environment_steps" or float(row["x"]) > args.max_step:
                continue
            baseline.setdefault(row["group"], []).append(
                (float(row["x"]), float(row["mean_return"]), float(row["std_return"]), int(row["n"]))
            )
    for group, rows in baseline.items():
        x, mean, std, n = (np.array(column) for column in zip(*rows))
        (line,) = ax.plot(x, mean, linestyle="--", linewidth=1.8, label=f"{group} (exp. 1, n={n[0]})")
        ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.15)

    # experiment 2: single seed runs, smoothed like the experiment 1 seeds
    for run, label in zip(args.runs, args.labels):
        _, rows = load_curve(run)
        data = np.array([row for row in rows if row[0] <= args.max_step])
        ax.plot(data[:, 0], smooth(data[:, 2], args.smooth_window), linewidth=2.2, label=f"{label} (exp. 2, n=1)")

    ax.set_xlim(0, args.max_step)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Evaluation episode return")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("R2-Dreamer: Reacher hard")
    fig.supxlabel(
        f"{args.smooth_window}-evaluation moving average per run; dashed: mean over seeds, shading: ±1 sample SD",
        fontsize=9,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output.with_suffix("." + extension), dpi=200)
    print("Saved", args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
