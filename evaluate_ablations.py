"""Evaluate a frozen image+state Reacher checkpoint under six input conditions."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import tools
from ablations import (
    CONDITIONS,
    PROPRIO_KEYS,
    STATE_KEYS,
    ObservationAblation,
    make_evaluation_env,
    policy_observation,
    sample_target,
)
from dreamer import Dreamer


@torch.inference_mode()
def rollout(
    agent, env_config, seed, condition="baseline", donor=None, record=False, time_limit=None, state_keys=PROPRIO_KEYS
):
    fake_target = sample_target((seed + 2_000_003) % 2**32) if condition == "wrong_target" else None
    env = make_evaluation_env(env_config, seed, fake_target, time_limit)
    started = time.perf_counter()
    try:
        obs = env.reset()
        initial = env.unwrapped._env.physics.get_state().copy()
        real_target = env.unwrapped.geometry()["target"]
        transform = ObservationAblation(condition, obs["image"], donor, state_keys)
        # Match stochastic RSSM draws across conditions; actor uses its evaluation mode.
        tools.set_seed_everywhere((seed + 3_000_017) % 2**32)
        state = agent.get_initial_state(1)
        frames, sequence, endpoint, true_proprio, input_proprio, actions = [], [], [], [], [], []
        score, step = 0.0, 0
        while True:
            sequence.append({key: obs[key].copy() for key in ("image", *STATE_KEYS)})
            fed = transform.apply(obs, step)
            if record:
                truth = env.unwrapped.render_true() if condition == "wrong_target" else obs["image"]
                frames.append(np.concatenate([truth, fed["image"]], axis=1))
                endpoint.append(env.unwrapped.geometry()["endpoint"])
                true_proprio.append(np.concatenate([obs[key] for key in state_keys]))
                input_proprio.append(np.concatenate([fed[key] for key in state_keys]))
            action, state = agent.act(policy_observation(fed, agent.device, state_keys), state, eval=True)
            action = action[0].float().cpu().numpy()
            if record:
                actions.append(action.copy())
            obs, reward, done, _ = env.step(action)
            score += float(reward)
            step += 1
            if done:
                # Donor streams include terminal observations for complete time alignment.
                sequence.append({key: obs[key].copy() for key in ("image", *STATE_KEYS)})
                break
        if agent.device.type == "cuda":
            torch.cuda.synchronize(agent.device)
        row = {
            "condition": condition,
            "seed": seed,
            "return": score,
            "environment_steps": step * int(env_config.action_repeat),
            "decision_steps": step,
            "seconds": time.perf_counter() - started,
            "real_target": real_target.tolist(),
            "fake_target": None if fake_target is None else fake_target.tolist(),
            "initial_state": initial.tolist(),
            "policy_seed": (seed + 3_000_017) % 2**32,
        }
        diagnostic = {
            "endpoint": np.asarray(endpoint),
            "true_proprio": np.asarray(true_proprio),
            "input_proprio": np.asarray(input_proprio),
            "actions": np.asarray(actions),
            "real_target": real_target,
            "fake_target": np.full(2, np.nan) if fake_target is None else fake_target,
        }
        return row, sequence, frames, diagnostic
    finally:
        env.close()


def load_agent(checkpoint_path, device):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if not all(key in saved for key in ("agent", "config")):
        raise ValueError("Expected a full training checkpoint containing config and agent state.")
    config = OmegaConf.create(saved["config"])
    if config.model.rep_loss != "r2dreamer":
        raise ValueError("This experiment expects an R2-Dreamer checkpoint.")

    # Saved configs are resolved, so update every nested device field explicitly.
    def replace_devices(node):
        if OmegaConf.is_dict(node):
            for key in node:
                if key == "device":
                    node[key] = str(device)
                elif OmegaConf.is_config(node[key]):
                    replace_devices(node[key])
        elif OmegaConf.is_list(node):
            for child in node:
                if OmegaConf.is_config(child):
                    replace_devices(child)

    replace_devices(config)
    config.model.compile = False
    env = make_evaluation_env(config.env, 0)
    try:
        agent = Dreamer(config.model, env.observation_space, env.action_space).to(device)
    finally:
        env.close()
    agent.load_state_dict(saved["agent"], strict=True)
    # image + joints (target only visible in the image) or image + full state (target also in to_target)
    allowed = (set(PROPRIO_KEYS), set(STATE_KEYS))
    if set(agent.encoder.cnn_shapes) != {"image"} or set(agent.encoder.mlp_shapes) not in allowed:
        raise ValueError("Checkpoint must encode the image plus position/velocity, optionally with to_target.")
    agent.requires_grad_(False)
    agent.eval()
    return agent, config, int(saved["trainer"]["step"])


def model_state_keys(agent):
    # state keys the model was trained on, in the order of STATE_KEYS
    return tuple(key for key in STATE_KEYS if key in agent.encoder.mlp_shapes)


def summarize(rows, bootstrap_seed=12345, samples=5000):
    rng = np.random.default_rng(bootstrap_seed)
    baseline = {row["seed"]: row["return"] for row in rows if row["condition"] == "baseline"}
    summary = []
    for condition in CONDITIONS:
        selected = sorted((row for row in rows if row["condition"] == condition), key=lambda row: row["seed"])
        if not selected:
            continue
        values = np.asarray([row["return"] for row in selected], dtype=float)
        delta = values - np.asarray([baseline[row["seed"]] for row in selected])
        indices = rng.integers(0, len(values), size=(samples, len(values)))

        def interval(array):
            return np.quantile(array[indices].mean(axis=1), [0.025, 0.975]).tolist()

        summary.append({
            "condition": condition,
            "episodes": len(values),
            "mean_return": float(values.mean()),
            "std_return": float(values.std(ddof=1)) if len(values) > 1 else None,
            "return_ci95": interval(values),
            "mean_paired_delta": float(delta.mean()),
            "paired_delta_ci95": interval(delta),
            "interval_note": "Episode bootstrap; conditional on this single trained checkpoint.",
        })
    return summary


def save_diagnostics(output, row, frames, diagnostic):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    directory = output / "examples"
    directory.mkdir(exist_ok=True)
    stem = f"{row['condition']}_seed{row['seed']}"
    np.savez_compressed(directory / f"{stem}.npz", **diagnostic)
    video = directory / f"{stem}.mp4"
    # Upscale nearest-neighbor so 64x64 inputs and labels are readable.
    with imageio.get_writer(video, fps=25, codec="libx264", macro_block_size=1) as writer:
        for frame in frames:
            image = Image.fromarray(frame).resize((frame.shape[1] * 3, frame.shape[0] * 3), Image.Resampling.NEAREST)
            canvas = Image.new("RGB", (image.width, image.height + 24))
            canvas.paste(image, (0, 24))
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 5), "Real scene", fill="white")
            draw.text((image.width // 2 + 4, 5), "Image fed to policy", fill="white")
            writer.append_data(np.asarray(canvas))
    row["example"] = str(video.relative_to(output))
    row["trajectory"] = str((directory / f"{stem}.npz").relative_to(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Training run directory or latest.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--videos", type=int, default=3, help="Recorded episodes per condition; 0 disables")
    parser.add_argument("--time-limit", type=int, help="Override saved episode length (for smoke tests only)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.videos < 0 or not 0 <= args.seed_start < 2**32 - args.episodes:
        parser.error("Require positive episodes, nonnegative videos, and valid 32-bit episode seeds.")
    os.environ.setdefault("MUJOCO_GL", "egl")
    checkpoint_path = args.run / "latest.pt" if args.run.is_dir() else args.run
    output = args.output_dir or checkpoint_path.parent / "ablations"
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently mixing or overwriting experiments.
    manifest_path = output / "experiment.json"
    with manifest_path.open("x") as stream:
        json.dump({"status": "initializing", "checkpoint": str(checkpoint_path.resolve())}, stream)
    device = torch.device(args.device)
    agent, config, training_step = load_agent(checkpoint_path, device)
    state_keys = model_state_keys(agent)
    print("State inputs fed to the policy:", state_keys)
    manifest = {
        "checkpoint": str(checkpoint_path.resolve()),
        "training_step": training_step,
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "device": str(device),
        "conditions": CONDITIONS,
        "state_inputs": state_keys,
        "modality_dropout": float(config.model.get("modality_dropout", 0.0)),
        "environment": OmegaConf.to_container(config.env, resolve=True),
        "time_limit_override": args.time_limit,
        "actor": "mode",
        "rssm": "sampled, matched policy seeds",
        "donor": "Independent baseline-policy episode; seed = live seed + 1000003 (mod 2**32)",
        "confidence_intervals": "95% paired episode bootstrap, not training-seed uncertainty",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    start = time.perf_counter()
    result_path = output / "episodes.jsonl"
    with result_path.open("x") as stream:
        for index in range(args.episodes):
            seed = args.seed_start + index
            donor_seed = (seed + 1_000_003) % 2**32
            _, donor, _, _ = rollout(
                agent, config.env, donor_seed, time_limit=args.time_limit, state_keys=state_keys
            )
            reference = None
            for condition in CONDITIONS:
                row, _, frames, diagnostic = rollout(
                    agent, config.env, seed, condition, donor, index < args.videos, args.time_limit, state_keys
                )
                matched = (row["initial_state"], row["real_target"])
                if reference is not None and matched != reference:
                    raise RuntimeError("Episode pairing failed: different initial state or real target.")
                reference = matched
                row["donor_seed"] = donor_seed if condition in ("unrelated", "corrupted_proprio") else None
                if frames:
                    save_diagnostics(output, row, frames, diagnostic)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                rows.append(row)
                print(
                    f"[{index + 1}/{args.episodes}] {condition}: return={row['return']:.1f}, "
                    f"steps={row['environment_steps']}, {row['seconds']:.2f}s",
                    flush=True,
                )
            if index == min(9, args.episodes - 1):
                elapsed = time.perf_counter() - start
                print(
                    f"Estimated remaining time: {elapsed / (index + 1) * (args.episodes-index-1) / 60:.1f} min",
                    flush=True,
                )
    summary = summarize(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "episodes.csv").open("w", newline="") as stream:
        fields = ("condition", "seed", "return", "environment_steps", "decision_steps", "seconds", "donor_seed")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    manifest["status"] = "complete"
    manifest["total_seconds"] = time.perf_counter() - start
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.no_plots:
        from plot_ablations import plot_results

        plot_results(output)
    print(f"Results saved in {output}")


if __name__ == "__main__":
    main()
