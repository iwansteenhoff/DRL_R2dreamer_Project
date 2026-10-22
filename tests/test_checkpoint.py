import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import checkpoint
from buffer import Buffer
from trainer import OnlineTrainer


def config():
    return OmegaConf.create({
        "logdir": "unused",
        "resume": False,
        "checkpoint": {"every": 10},
        "env": {"steps": 100, "task": "reacher"},
        "trainer": {"steps": 100, "train_ratio": 1},
    })


def replay(directory=None):
    return Buffer(
        SimpleNamespace(
            device="cpu",
            storage_device="cpu",
            backend="memmap" if directory else "memory",
            directory=str(directory),
            batch_size=1,
            batch_length=2,
            max_size=30,
            online=True,
            update_latents=False,
        )
    )


def agent():
    model = torch.nn.Linear(2, 1)
    model.device = torch.device("cpu")
    model._optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model._scheduler = torch.optim.lr_scheduler.StepLR(model._optimizer, 1)
    model._slow_value_updates = 3
    model(torch.ones(1, 2)).sum().backward()
    model._optimizer.step()
    model._scheduler.step()
    return model


@pytest.mark.parametrize("backend", ["memory", "memmap"])
def test_full_roundtrip_and_atomic_failure(tmp_path, backend):
    directory = tmp_path / "replay" if backend == "memmap" else None
    model, buffer = agent(), replay(directory)
    for i in range(7):
        buffer.add_transition(
            TensorDict(
                {
                    "value": torch.tensor([[float(i)]]),
                    "episode": torch.tensor([0]),
                    "action": torch.zeros(1, 1),
                    "stoch": torch.zeros(1, 1),
                    "deter": torch.zeros(1, 1),
                },
                batch_size=[1],
            )
        )
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text('{"step": 12}\n')
    trainer = SimpleNamespace(
        logger=SimpleNamespace(_logdir=tmp_path),
        replay_buffer=buffer,
        started_at=0.0,
        step=12,
        state_dict=lambda: {"step": 12, "active_seconds": 25, "curve": []},
    )
    path = tmp_path / "latest.pt"
    checkpoint.save(path, checkpoint.experiment_config(config()), model, trainer)
    expected_rng = (random.random(), np.random.rand(), torch.rand(2))
    loaded = checkpoint.load(path, config())
    new_model, new_buffer = agent(), replay(directory)
    checkpoint.restore(loaded, new_model, new_buffer)
    assert random.random() == expected_rng[0]
    assert np.random.rand() == expected_rng[1]
    assert torch.equal(torch.rand(2), expected_rng[2])
    assert new_buffer.count() == buffer.count()
    assert not new_buffer._online._pending
    assert len(new_buffer._online.queue) == len(buffer._online.queue)
    data, _, _ = new_buffer.sample()
    assert data["value"].shape == (1, 2, 1)
    assert new_model._scheduler.last_epoch == model._scheduler.last_epoch
    assert new_model._slow_value_updates == 3
    for key, value in model.state_dict().items():
        assert torch.equal(new_model.state_dict()[key], value)
    for state in new_model._optimizer.state.values():
        assert state["step"].item() == 1
    assert loaded["metrics_bytes"] == metrics.stat().st_size
    before = path.read_bytes()
    with patch("checkpoint.torch.save", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            checkpoint.save(path, checkpoint.experiment_config(config()), model, trainer)
    assert path.read_bytes() == before


def test_configuration_validation(tmp_path):
    original = config()
    extended = config()
    extended.env.steps = extended.trainer.steps = 200
    extended.resume = True
    assert checkpoint.experiment_config(original) == checkpoint.experiment_config(extended)
    original.trainer.update_log_every = 5000
    extended.trainer.update_log_every = 1000
    assert checkpoint.experiment_config(original) == checkpoint.experiment_config(extended)
    extended.trainer.train_ratio = 2
    assert checkpoint.experiment_config(original) != checkpoint.experiment_config(extended)


def test_clock_excludes_downtime_and_curve_rollback(tmp_path):
    trainer = OnlineTrainer.__new__(OnlineTrainer)
    trainer.logger = SimpleNamespace(_logdir=tmp_path)
    trainer.elapsed_seconds, trainer.started_at = 25.0, 1000.0
    with patch("trainer.time.monotonic", return_value=1004.0):
        assert trainer.active_seconds() == 29.0
    trainer.curve = [{"step": 10, "active_seconds": 25.0, "eval_score": 1.0}]
    (tmp_path / "learning_curve.jsonl").write_text("discard uncheckpointed evaluation\n")
    trainer.write_curve()
    import json

    assert [json.loads(row) for row in (tmp_path / "learning_curve.jsonl").read_text().splitlines()] == trainer.curve


def test_workers_survive_parent_handled_interrupts():
    import os
    import signal

    from envs.parallel import ProcessPipeWorker

    worker = ProcessPipeWorker(lambda state, value: (state, value + 1))
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            os.kill(worker._process.pid, sig)
            assert worker(2)() == 3
    finally:
        worker.close()
