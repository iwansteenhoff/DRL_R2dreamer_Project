"""Compare image-only and image+proprioception means and standard deviations across seeds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from plot_learning_curves import load_curve


def aggregate(groups, coordinate):
    """Interpolate onto a shared grid, restricting it to every run's observed range."""
    curves = {}
    for label, runs in groups.items():
        curves[label] = []
        for run in runs:
            data = np.asarray(run, dtype=float)
            x, y = data[:, coordinate], data[:, 2]
            # If time measurements coincide, retain the later evaluation.
            keep = np.r_[np.diff(x) != 0, True]
            curves[label].append((x[keep], y[keep]))
    all_curves = [curve for runs in curves.values() for curve in runs]
    lower = max(x[0] for x, _ in all_curves)
    upper = min(x[-1] for x, _ in all_curves)
    if lower >= upper:
        raise ValueError("Runs do not have an overlapping observation range.")
    grid = np.unique(np.concatenate([x[(x >= lower) & (x <= upper)] for x, _ in all_curves]))
    result = {}
    for label, runs in curves.items():
        aligned = np.stack([np.interp(grid, x, y) for x, y in runs])
        result[label] = {"mean": aligned.mean(axis=0), "std": aligned.std(axis=0, ddof=1), "n": len(runs)}
    return grid, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", nargs="+", default=[f"logdir/reacher_hard_images_cpu_replay_seed{seed}" for seed in range(3)]
    )
    parser.add_argument(
        "--proprio",
        nargs="+",
        default=[f"logdir/reacher_hard_images_proprio_cpu_replay_seed{seed}" for seed in range(3)],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("plots/seed_comparison"))
    args = parser.parse_args()
    groups, sources = {}, {}
    for label, paths in (("Images only", args.images), ("Images + proprioception", args.proprio)):
        if len(paths) < 2:
            parser.error("Each group needs at least two runs for a sample standard deviation.")
        if len(set(str(Path(path).resolve()) for path in paths)) != len(paths):
            parser.error("Each group must contain distinct runs.")
        loaded = [load_curve(path) for path in paths]
        groups[label] = [rows for _, rows in loaded]
        sources[label] = [str(path.resolve()) for path, _ in loaded]
        print(f"{label}: {len(paths)} runs")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True, layout="constrained")
    records = []
    ranges = {}
    for axis, coordinate, name, scale in ((axes[0], 0, "environment_steps", 1), (axes[1], 1, "active_hours", 3600)):
        grid, stats = aggregate(groups, coordinate)
        ranges[name] = [float(grid[0] / scale), float(grid[-1] / scale)]
        for label, values in stats.items():
            mean, std = values["mean"], values["std"]
            (line,) = axis.plot(grid / scale, mean, label=f"{label} (n={values['n']})", linewidth=2)
            axis.fill_between(grid / scale, mean - std, mean + std, color=line.get_color(), alpha=0.2)
            records.extend(
                {
                    "axis": name,
                    "x": float(x / scale),
                    "group": label,
                    "mean_return": float(m),
                    "std_return": float(s),
                    "n": values["n"],
                }
                for x, m, s in zip(grid, mean, std)
            )
        axis.grid(alpha=0.25)
        axis.set_xlim(left=0)
        axis.legend(frameon=False)
    axes[0].set_title("Return versus environment steps")
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Evaluation episode return")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    axes[0].xaxis.set_major_formatter(formatter)
    axes[1].set_title("Return versus active wall-clock time")
    axes[1].set_xlabel("Cumulative active time (hours)")
    fig.suptitle("R2-Dreamer: Reacher hard")
    fig.supxlabel("Lines: mean across training seeds; shaded bands: ±1 sample standard deviation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = args.output_dir / f"mean_std_comparison.{extension}"
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    plt.close(fig)
    with (args.output_dir / "mean_std_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "sources": sources,
        "std_ddof": 1,
        "interpolation": "linear, no extrapolation or smoothing",
        "shared_ranges": ranges,
        "time_definition": "Active time excludes setup, downtime and checkpoint saving.",
    }
    (args.output_dir / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
