"""Full training checkpoints. Load only checkpoints from trusted runs."""

import copy
import os
import random
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from disk_replay import sync_directory

_COUNTERS = ("_slow_value_updates", "_ema_updates", "_nf_tensors", "_nf_skipped", "_nf_streak")


def experiment_config(config):
    # Operational settings may change; the experiment and replay layout may not.
    state = OmegaConf.to_container(config, resolve=True)
    for key in ("logdir", "resume", "checkpoint"):
        state.pop(key, None)
    state["env"].pop("steps", None)
    state["trainer"].pop("steps", None)
    state["trainer"].pop("update_log_every", None)
    if "buffer" in state:
        for key in ("backend", "directory", "storage_device"):
            state["buffer"].pop(key, None)
    return state


def load(path, config):
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if state.get("version") not in (1, 2):
        raise ValueError("This checkpoint does not contain full resumable training state.")
    if experiment_config(OmegaConf.create(state["config"])) != experiment_config(config):
        raise ValueError("Resume configuration differs from the checkpoint experiment configuration.")
    return state


def restore(state, agent, replay):
    agent.load_state_dict(state["agent"])
    # Detach small optimizer tensors from the checkpoint's whole-file mmap.
    agent._optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    agent._scheduler.load_state_dict(state["scheduler"])
    for name, value in state["counters"].items():
        setattr(agent, name, value)
    replay.load_state_dict(state["replay"])
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save(path, config_state, agent, trainer):
    # Exclude checkpoint I/O consistently from active training time.
    if agent.device.type == "cuda":
        torch.cuda.synchronize(agent.device)
    started = time.monotonic()
    state = {
        "version": 2,
        "config": config_state,
        "metrics_bytes": (
            (trainer.logger._logdir / "metrics.jsonl").stat().st_size
            if (trainer.logger._logdir / "metrics.jsonl").exists()
            else 0
        ),
        "agent": agent.state_dict(),
        "optimizer": agent._optimizer.state_dict(),
        "scheduler": agent._scheduler.state_dict(),
        "counters": {name: getattr(agent, name) for name in _COUNTERS if hasattr(agent, name)},
        "replay": trainer.replay_buffer.state_dict(),
        "trainer": trainer.state_dict(),
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
        trainer.replay_buffer.checkpoint_committed(state["replay"])
    finally:
        trainer.started_at += time.monotonic() - started
    duration = time.monotonic() - started
    size_mb = path.stat().st_size / 1024**2
    print(f"Saved checkpoint at step {trainer.step}: {path} ({size_mb:.1f} MiB, {duration:.2f}s)")
