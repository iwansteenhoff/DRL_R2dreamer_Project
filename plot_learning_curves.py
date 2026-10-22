"""Plot evaluation return against environment steps and cumulative active time."""

import argparse
import json
import math
from pathlib import Path


def load_curve(path):
    path = Path(path)
    if path.is_dir():
        path = path / "learning_curve.jsonl"
    rows = []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                values = tuple(float(row[key]) for key in ("step", "active_seconds", "eval_score"))
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("non-finite value")
                if values[0] < 0 or values[1] < 0:
                    raise ValueError("negative step or time")
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid curve record at {path}:{number}: {error}") from error
            if rows and (values[0] <= rows[-1][0] or values[1] < rows[-1][1]):
                raise ValueError(f"Steps must increase and active time must not decrease: {path}:{number}")
            rows.append(values)
    if not rows:
        raise ValueError(f"No evaluation records in {path}")
    return path, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run directories or learning_curve.jsonl files")
    parser.add_argument("--labels", nargs="+", help="One legend label per run, in the same order")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument("--title", default="R2-Dreamer learning curves")
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.runs):
        parser.error("--labels must contain exactly one label per run")

    # Render files without requiring a display on the university workstation.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except ImportError:
        parser.error('Install plotting dependencies with: python -m pip install -e ".[plot]"')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
    for index, run in enumerate(args.runs):
        try:
            path, rows = load_curve(run)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        steps, seconds, scores = zip(*rows)
        label = args.labels[index] if args.labels else path.parent.name
        (line,) = axes[0].plot(steps, scores, label=label, linewidth=1.6)
        axes[1].plot([value / 3600 for value in seconds], scores, label=label, color=line.get_color(), linewidth=1.6)
        print(
            f"{label}: {len(rows)} evaluations; last evaluation at {steps[-1]:,.0f} steps, "
            f"{seconds[-1] / 3600:.2f} active hours"
        )

    axes[0].set_title("Return versus environment steps")
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Evaluation episode return")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    axes[0].xaxis.set_major_formatter(formatter)
    axes[1].set_title("Return versus active wall-clock time")
    axes[1].set_xlabel("Cumulative active time (hours)")
    for axis in axes:
        axis.set_xlim(left=0)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle(args.title)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        output = args.output_dir / f"learning_curves.{extension}"
        fig.savefig(output, dpi=200)
        print(f"Saved {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
