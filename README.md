## R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation

This repository provides the official implementation of [R2-Dreamer][r2dreamer] (ICLR 2026), a decoder-free, augmentation-free world model with redundancy-reducing regularization on latent representations. It also includes an efficient PyTorch DreamerV3 reproduction that trains **~5x faster** than a widely used [codebase][dreamerv3-torch], along with other baselines. Selecting R2-Dreamer via the config provides an additional **~1.6x speedup** over this baseline.

## Instructions

Install dependencies. This repository is tested with Ubuntu 24.04 and Python 3.11.

Installing via a virtual env like [uv](https://docs.astral.sh/uv/) is recommended. If you prefer Docker, follow [`docs/docker.md`](docs/docker.md).

```bash
# Select the environments you need, or use "all" to install everything.
# Available extras: dmc, atari, crafter, metaworld, memorymaze, isaaclab, all
pip install -e ".[dmc,metaworld]"
```

Run training on default settings:

```bash
python3 train.py logdir=./logdir/test
```

Monitoring results:
```bash
tensorboard --logdir ./logdir
```

Switching algorithms:

```bash
# Choose an algorithm via model.rep_loss:
# r2dreamer|dreamer|infonce|dreamerpro
python3 train.py model.rep_loss=r2dreamer
```

For easier code reading, inline tensor shape annotations are provided. See [`docs/tensor_shapes.md`](docs/tensor_shapes.md).


## Available Benchmarks
At the moment, the following benchmarks are available in this repository.

| Environment        | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [Meta-World](https://github.com/Farama-Foundation/Metaworld) | Image | Continuous | 1M | Robotic manipulation with complex contact interactions.|
| [DMC Proprio](https://github.com/deepmind/dm_control) | State | Continuous | 500K | DeepMind Control Suite with low-dimensional inputs. |
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous |1M| DeepMind Control Suite with high-dimensional images inputs. |
| [DMC Subtle](envs/dmc_subtle.py) | Image | Continuous |1M| DeepMind Control Suite with tiny task-relevant objects. |
| [Atari 100k](https://github.com/Farama-Foundation/Arcade-Learning-Environment) | Image | Discrete |400K| 26 Atari games. |
| [Crafter](https://github.com/danijar/crafter) | Image | Discrete |1M| Survival environment to evaluates diverse agent abilities.|
| [Memory Maze](https://github.com/jurgisp/memory-maze) | Image |Discrete |100M| 3D mazes to evaluate RL agents' long-term memory.|
| [IsaacLab](https://github.com/isaac-sim/IsaacLab) | State / Image | Continuous | 0.5M–1M | GPU-parallelized robotic simulation. |

Use Hydra to select a benchmark and a specific task using `env` and `env.task`, respectively.

```bash
python3 train.py ... env=dmc_vision env.task=dmc_walker_walk
```

## Headless rendering

If you run MuJoCo-based environments (DMC / MetaWorld) on headless machines, you may need to set `MUJOCO_GL` for offscreen rendering. **Using EGL is recommended** as it accelerates rendering, leading to faster simulation throughput.

```bash
# For example, when using EGL (GPU)
export MUJOCO_GL=egl
# (optional) Choose which GPU EGL uses
export MUJOCO_EGL_DEVICE_ID=0
```

More details: [Working with MuJoCo-based environments](https://docs.pytorch.org/rl/stable/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)

## Code formatting

If you want automatic formatting/basic checks before commits, you can enable `pre-commit`:

```bash
pip install pre-commit
# This sets up a pre-commit hook so that checks are run every time you commit
pre-commit install
# Manual pre-commit run on all files
pre-commit run --all-files
```

## Troubleshooting

If you encounter installation or runtime issues, please refer to our [Troubleshooting Guide](docs/troubleshooting.md).

## Citation

If you find this code useful, please consider citing:

```bibtex
@inproceedings{
morihira2026rdreamer,
title={R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation},
author={Naoki Morihira and Amal Nahar and Kartik Bharadwaj and Yasuhiro Kato and Akinobu Hayashi and Tatsuya Harada},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Je2QqXrcQq}
}
```

[r2dreamer]: https://openreview.net/forum?id=Je2QqXrcQq&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)
[dreamerv3-torch]: https://github.com/NM512/dreamerv3-torch

## Interrupted runs and learning curves

Replay is stored continuously in memory-mapped files under `<logdir>/replay/`
on the local disk. Training keeps its model and sampled batches on the configured
device (normally CUDA), while replay storage uses CPU-accessible file mappings.
The replay directory can reach approximately 11 GB with the default model and
500,000-transition capacity. The OS manages its cached pages; the entire replay
does not need to remain resident in RAM.

Training metrics print approximately every **1,000 environment steps** once updates
begin (`trainer.update_log_every=1000`).

Training saves model/optimizer state, replay metadata, the bounded online queue,
and learning-curve history to `<logdir>/latest.pt` every **50,000 environment
steps** and at normal completion. Change this with `checkpoint.every=20000`;
`checkpoint.every=0` disables periodic saves. These checkpoints do not include
the replay's image/latent arrays. They are typically hundreds of MB rather than
11 GB; exact size depends on the model and online queue. Replay is flushed before
checkpoint replacement; this still costs disk I/O but avoids repeatedly serializing
all replay images and unused capacity.

Ctrl+C or SIGTERM requests a save at the end of the current training iteration.
A hard kill or machine failure recovers from the last completed checkpoint.
Model checkpoints are replaced atomically, temporarily keeping the old and new
small files. Once committed replay rows start being overwritten (or latent states
are updated), an undo journal records their previous values before mutation.
Recovery reverts those changes and discards appended entries beyond the saved
length, so model state and replay both return to the checkpoint. The journal is
cleared after a successful checkpoint. Its size depends on how much committed
replay changes between checkpoints; in the worst case it can approach replay
size. With the default disabled latent updates it is empty until circular writes
reach entries already present at the last checkpoint.

**Keep `latest.pt` and the entire `replay/` directory together.** The small
checkpoint alone cannot recover the images. Use a separate log directory for
each experiment; concurrent training writers in one directory are rejected.
`/local` is recommended on LIACS workstations; its files are specific to that host.
Use `buffer.backend=memory` for the original full-checkpoint implementation.

Resume with the **same experiment overrides and log directory**, adding
`resume=True`, for example:

```bash
python train.py env=dmc_vision env.task=dmc_reacher_easy \
  seed=0 logdir=./logdir/reacher_images_seed0 resume=True
```

Model, optimizer, learning-rate scheduler, replay (including ready online
windows), update counters, scheduling state, and Python/NumPy/PyTorch random
states are restored. The environment-step budget remains a total budget, not
an additional number of steps. It can be increased with `env.steps=...`.
Other experiment settings are validated against the checkpoint. Existing full
checkpoints from the earlier implementation convert automatically to disk replay
when resumed with `buffer.backend=memmap` (the default). Conversion copies only
populated rows and preserves the original checkpoint until the next successful
save. This first conversion temporarily needs space for both the original file
and replay files; later saves are small. Older weights-only `latest.pt` files
cannot resume a complete run. Load only your own trusted
checkpoints, since full replay recovery uses Python object deserialization.

Simulator episodes restart on resume; pending online windows and unfinished
training-episode statistics are discarded. New replay sequences have distinct
identifiers so samples cannot span the interruption. Simulator random states
are not restored, so continuation is not bit-for-bit identical to an uninterrupted
run.

`<logdir>/learning_curve.jsonl` contains one record per evaluation:

```json
{"step": 10000, "active_seconds": 123.4, "eval_score": 456.7}
```

Plot `eval_score` against `step` for an environment-steps learning curve, or
against `active_seconds` for a wall-clock learning curve. `step` counts environment
steps including action repeats, summed over training environments, and excludes
evaluation steps. The time coordinate is sampled **before** evaluation, matching
the policy being evaluated. Active time accumulates across resumed sessions;
it includes environment interaction, model updates, logging, compilation during
the loop, and previous evaluations. It excludes process/model/environment setup,
checkpoint flushing/serialization, and downtime between sessions. Use identical evaluation
and checkpoint settings across comparison runs. Enable evaluation
(`env.eval_episode_num>0`) to obtain these curve records.

The same coordinates are logged in `metrics.jsonl` and TensorBoard
(`episode/eval_score` and `time/active_seconds`). On resume, uncheckpointed curve
and JSON metric records are rolled back, and TensorBoard receives a purge marker
for steps beyond the checkpoint. The checkpoint retains the complete evaluation
curve, so it remains recoverable even if the curve file was interrupted during
writing.


## Plotting learning curves

Install the optional plotting dependency and generate a two-panel PNG and PDF:

```bash
python -m pip install -e ".[plot]"
python plot_learning_curves.py logdir/reacher_hard_images_cpu_replay_seed0 \
  --labels "Images only" --title "R2-Dreamer: Reacher hard"
```

Outputs are `plots/learning_curves.png` and `plots/learning_curves.pdf`. The left
panel plots evaluation episode return against environment steps; the right uses
cumulative active time in hours (excluding setup, downtime, and checkpoint saves).
The script displays raw evaluations without smoothing or confidence intervals;
a single run cannot quantify variability across training seeds. The final plotted
point is the last evaluation, which can precede the final training step.

To compare modalities later, pass both run directories followed by matching labels:

```bash
python plot_learning_curves.py logdir/images_run logdir/images_proprio_run \
  --labels "Images only" "Images + proprioception" --output-dir plots/comparison
```


## Evaluating visual and proprioceptive reliance

Run this **after image + proprioception training finishes**, inside tmux with the
project environment activated. It loads the checkpoint's saved architecture and
DMC settings; you do not need to repeat training overrides. No replay is loaded,
no optimizer updates run, and model parameters are frozen. Install `.[plot]` if
matplotlib is not already available (as in the plotting instructions above).

```bash
MUJOCO_GL=egl python evaluate_ablations.py \
  logdir/reacher_hard_images_proprio_cpu_replay_seed0 \
  --episodes 100 --seed-start 100000 --videos 3
```

The default device is `cuda:0`; use `--device cpu` for CPU inference. Each of the
100 matched episode seeds is evaluated in all six conditions:

| Condition | Images | Joint position and velocity |
|---|---|---|
| `baseline` | Live, correct scene | Live, correct |
| `wrong_target` | Current arm, independently positioned visible target | Live, correct |
| `frozen` | First image of the current episode, repeated | Live, correct |
| `unrelated` | Time-aligned images from an independent donor episode | Live, correct |
| `black` | All-zero uint8 images | Live, correct |
| `corrupted_proprio` | Live, correct scene | Time-aligned joint states from the donor episode |

Donors are independent baseline-policy episodes from this same frozen model,
not random vectors or random-action trajectories. One donor is recorded per
matched seed and shared by `unrelated` and `corrupted_proprio`. Donor collection
adds 100 episodes, so a full run executes **700 episodes** in total. A timing
estimate prints after the first ten matched seeds. Evaluation runs sequentially,
uses actor mode actions and stochastic RSSM inference, and resets the policy RNG
to the same seed before each matched condition. It verifies identical initial
physics states and real target coordinates across conditions. Only image,
position, velocity, and episode flags reach the policy; target vectors and rewards
are never fed to it.

Fake targets are sampled independently from Reacher's normal angle/radius
distribution and held fixed during the episode. Only MuJoCo's rendered scene is
changed via `scene_callback`; simulation/model geometry, real targets, state
observations, and reward calculations remain untouched. The fake target can
occasionally land close to the real target, as expected under independent sampling.
The first frozen frame still contains the real target; that condition tests loss
of visual feedback, not complete removal of all target information. Donor inputs
preserve familiar ranges but deliberately break consistency with the live episode.
Black images remain an out-of-distribution stress test. None of these ablations
alone proves the model completely ignores or exclusively uses a modality.

Results default to `<run>/ablations/`:

- `experiment.json`: checkpoint path/training step, device, seed and condition definitions.
- `episodes.jsonl` and `episodes.csv`: individual returns, episode lengths, timings,
  and seeds. JSONL also records initial states and true/fake target coordinates.
- `summary.json`: mean returns and paired differences from normal inputs, with
  reproducible 95% bootstrap intervals over evaluation episodes.
- `ablation_comparison.png` / `.pdf`: episode points, means, return intervals,
  and paired return differences.
- `examples/`: side-by-side videos (real scene on the left; actual image fed to
  the policy on the right), plus endpoint trajectories, true/input joint states,
  and actions in NPZ files. Three episodes per condition are recorded by default.
- `example_trajectories.png` / `.pdf`: first matched example, showing endpoint paths
  and real/fake targets. Videos show actions before each transition at 25 FPS,
  appropriate for standard Reacher with action repeat 2.

Intervals describe uncertainty over evaluation episodes **for this one trained
checkpoint**, not variability across independently trained models. No smoothing
or fabricated training-seed uncertainty is added. For corrupted proprioception
the video image panels are identical; its NPZ file records the changed joint inputs.

Use `--videos 0` to disable videos/trajectories and `--no-plots` to skip plotting.
Replot saved results with:

```bash
python plot_ablations.py logdir/reacher_hard_images_proprio_cpu_replay_seed0/ablations
```

Existing output directories containing `experiment.json` are never overwritten;
choose a new `--output-dir` when repeating an experiment. Per-episode JSONL records
are flushed as evaluation proceeds. A quick functionality check (not a scientific
result) can shorten episodes:

```bash
MUJOCO_GL=egl python evaluate_ablations.py \
  logdir/reacher_hard_images_proprio_cpu_replay_seed0 \
  --episodes 2 --time-limit 8 --videos 0 --output-dir /tmp/reacher_ablation_check
```

For the real experiment, omit `--time-limit` to retain the checkpoint's original
1,000-environment-step episodes.

To compare the three training seeds (0, 1, 2) per modality with mean ±1 sample
standard deviation, run:

```bash
python plot_seed_comparison.py
```

This reads the six `reacher_hard_images[_proprio]_cpu_replay_seed*` runs and saves
`plots/seed_comparison/mean_std_comparison.png` and `.pdf`, plus the aggregated
values in `mean_std_data.csv` and source/alignment information in
`plot_metadata.json`. Both panels use linear interpolation onto shared grids
within the range observed by **all six runs**, with no extrapolation. Each seed
is first smoothed with a centered five-evaluation moving average (shorter windows
at boundaries); the mean and SD are computed from those smoothed seed curves.
Use `--smooth-window 1` for raw curves, or another positive odd window size.
The time panel therefore stops at the shortest run's last recorded active time.
Bands show sample standard deviation across three training seeds (`ddof=1`),
not a confidence interval or variation across evaluation episodes. Override
`--images`, `--proprio`, and `--output-dir` to use different run groups or paths.


## Experiment 2: images plus a state that is enough on its own

The experiment above feeds the policy joint positions and velocities, so the target is only visible in
the image. In this second experiment the encoder also receives `to_target`, the vector from the fingertip
to the target. The state alone is then enough to solve Reacher, which lets us check whether the model
still uses the images when it does not need them.

`configs/env/dmc_multimodal.yaml` sends the image to the CNN and `position`, `to_target` and `velocity`
to the MLP. Budget and settings match `dmc_vision`, so the existing image-only Reacher hard run is
the baseline and is not trained again.

Modality dropout is set with `model.modality_dropout` (default 0, off). With probability p a training
sequence loses one modality for the whole window, the image or the state with equal chance, never both.
The lost part of the encoder embedding is set to zero before it enters the posterior, so the world model
has to infer the latent state from the remaining modality. The R2-Dreamer redundancy loss still uses the
full embedding as its target. Dropout is only applied during training updates, never while acting
or evaluating. With p = 0.5, a quarter of the sequences lose the image, a quarter lose the state, and
half see both.

The run script trains image + state with and without dropout and then evaluates both checkpoints:

```bash
bash runs/reacher_state_sufficient.sh [seed] [env steps]   # defaults: seed 0, 5.1e5 steps
```

Running the script again with the same seed and a bigger budget continues both runs from their
checkpoints (for example `bash runs/reacher_state_sufficient.sh 0 1.01e6`). The step budget only
decides when the loop stops, so a continued run trains the same way as one started with the bigger budget.
Each budget writes its ablations to its own `ablations_steps<budget>/` folder.

`evaluate_ablations.py` detects which state inputs a checkpoint uses and feeds only those to the policy.
For these models `corrupted_proprio` replaces all three state inputs with the donor episode, including
`to_target`, so in that condition the true target is only visible in the image. In `wrong_target` the
image shows a fake target while `to_target` still points at the real one, so the two modalities
disagree. The experiment manifest records `state_inputs` and `modality_dropout`.
