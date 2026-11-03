"""Aggregate ablations across trained checkpoints without treating episodes as independent training runs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from plot_ablations import LABELS


def load_runs(directories):
    conditions = tuple(LABELS)
    runs, manifests, expected_seeds, model_seeds = [], [], None, set()
    for directory in directories:
        directory = Path(directory)
        manifest = json.loads((directory / "experiment.json").read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"Incomplete evaluation: {directory}")
        model_seed = manifest["environment"]["seed"]
        if model_seed in model_seeds:
            raise ValueError(f"Repeated training seed: {model_seed}")
        model_seeds.add(model_seed)
        rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines() if line.strip()]
        indexed = {}
        for row in rows:
            key = (row["condition"], row["seed"])
            if key in indexed:
                raise ValueError(f"Duplicate condition/episode: {directory}: {key}")
            indexed[key] = row
        seeds = sorted(seed for condition, seed in indexed if condition == "baseline")
        if len(seeds) != manifest["episodes"]:
            raise ValueError(f"Missing baseline episodes: {directory}")
        if expected_seeds is None:
            expected_seeds = seeds
        if seeds != expected_seeds or set(indexed) != {(condition, seed) for condition in conditions for seed in seeds}:
            raise ValueError(f"Conditions/episode seeds do not match: {directory}")
        if manifests:
            reference = manifests[0]
            for key in ("training_step", "time_limit_override", "actor", "rssm"):
                if manifest.get(key) != reference.get(key):
                    raise ValueError(f"Evaluation settings differ ({key}): {directory}")
            for key in ("task", "action_repeat", "time_limit", "size", "encoder"):
                if manifest["environment"][key] != reference["environment"][key]:
                    raise ValueError(f"Environment settings differ ({key}): {directory}")
        for seed in seeds:
            baseline = indexed[("baseline", seed)]
            for condition in conditions:
                row = indexed[(condition, seed)]
                if (row["initial_state"], row["real_target"]) != (baseline["initial_state"], baseline["real_target"]):
                    raise ValueError(f"Unmatched initial state/target: {directory}, seed {seed}")
                if row["environment_steps"] != baseline["environment_steps"]:
                    raise ValueError(f"Episode lengths differ: {directory}, seed {seed}")
        values = np.asarray([[indexed[(condition, seed)]["return"] for seed in seeds] for condition in conditions])
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite returns: {directory}")
        runs.append(values)
        manifests.append(manifest)
    return np.stack(runs), manifests, expected_seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, default=[
        Path(f"logdir/reacher_hard_images_proprio_cpu_replay_seed{seed}/ablations_all_seeds") for seed in range(5)])
    parser.add_argument("--output-dir", type=Path, default=Path("plots/five_seed_ablations"))
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("At least two trained checkpoints are required.")
    try:
        values, manifests, episode_seeds = load_runs(args.runs)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    # First average episodes within each trained model, then summarize across models.
    seed_means = values.mean(axis=2)
    seed_deltas = (values - values[:, :1, :]).mean(axis=2)
    summary, per_seed = [], []
    for i, condition in enumerate(LABELS):
        means, deltas = seed_means[:, i], seed_deltas[:, i]
        summary.append({"condition": condition, "training_seeds": len(values),
                        "episodes_per_training_seed": len(episode_seeds),
                        "mean_return": float(means.mean()), "std_across_training_seeds": float(means.std(ddof=1)),
                        "mean_paired_delta": float(deltas.mean()),
                        "paired_delta_std_across_training_seeds": float(deltas.std(ddof=1))})
        for j, manifest in enumerate(manifests):
            per_seed.append({"training_seed": manifest["environment"]["seed"], "condition": condition,
                             "mean_return": float(means[j]), "mean_paired_delta": float(deltas[j])})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("summary", summary), ("per_seed_means", per_seed)):
        (args.output_dir / f"{name}.json").write_text(json.dumps(records, indent=2) + "\n")
        with (args.output_dir / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    metadata = {"sources": [str(path.resolve()) for path in args.runs], "episode_seeds": episode_seeds,
                "training_seeds": [m["environment"]["seed"] for m in manifests], "training_step": manifests[0]["training_step"],
                "method": "Equal weight per trained checkpoint; mean and sample SD (ddof=1) of per-model episode means.",
                "paired_deltas": "Subtract baseline return within each matching episode and trained checkpoint.",
                "uncertainty": "Error bars are across-training-seed SD, not episode SD or confidence intervals."}
    (args.output_dir / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), layout="constrained")
    offsets = np.linspace(-.16, .16, len(values))
    for axis, data, title, ylabel in (
        (axes[0], seed_means, "Evaluation return", "Mean episode return"),
        (axes[1], seed_deltas, "Paired change from normal inputs", "Return difference (condition − baseline)"),
    ):
        x = np.arange(len(LABELS))
        axis.bar(x, data.mean(axis=0), color="tab:blue", alpha=.2, width=.6)
        axis.errorbar(x, data.mean(axis=0), yerr=data.std(axis=0, ddof=1), fmt="ko", capsize=5,
                      linewidth=1.5, label="Mean ±1 SD across training seeds", zorder=4)
        for i, manifest in enumerate(manifests):
            axis.scatter(x + offsets[i], data[i], s=22, label=f"Training seed {manifest['environment']['seed']}",
                         color=f"C{i % 10}", zorder=3)
        axis.set_xticks(x, list(LABELS.values()), rotation=30, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.25)
        axis.axhline(0, color="gray", linewidth=.8)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Image + proprioception ablations: {len(values)} trained seeds, {len(episode_seeds)} matched episodes each")
    fig.supxlabel("Dots: each trained model's episode mean; black: across-model mean ±1 sample standard deviation")
    for extension in ("png", "pdf"):
        path = args.output_dir / f"ablation_comparison.{extension}"
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    plt.close(fig)
    for row in summary:
        print(f"{row['condition']}: {row['mean_return']:.2f} ± {row['std_across_training_seeds']:.2f}")


if __name__ == "__main__":
    main()
