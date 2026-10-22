import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from ablations import (
    CONDITIONS,
    ObservationAblation,
    make_evaluation_env,
    policy_observation,
    sample_target,
)
from evaluate_ablations import summarize


@pytest.fixture(autouse=True)
def headless_rendering(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))


def observation(value):
    return {
        "image": np.full((8, 8, 3), value, np.uint8),
        "position": np.full(2, value, np.float32),
        "velocity": np.full(2, value + 0.5, np.float32),
        "to_target": np.array([99.0, 99.0]),
        "is_first": True,
        "is_last": False,
        "is_terminal": False,
    }


@pytest.mark.parametrize("condition", CONDITIONS)
def test_corruptions_modify_only_the_intended_channels(condition):
    initial, live, donor = observation(1), observation(2), [observation(3)]
    transform = ObservationAblation(condition, initial["image"], donor)
    actual = transform.apply(live, 0)
    if condition == "black":
        assert np.all(actual["image"] == 0)
    elif condition == "frozen":
        np.testing.assert_array_equal(actual["image"], initial["image"])
    elif condition == "unrelated":
        np.testing.assert_array_equal(actual["image"], donor[0]["image"])
    else:
        np.testing.assert_array_equal(actual["image"], live["image"])
    for key in ("position", "velocity"):
        np.testing.assert_array_equal(actual[key], donor[0][key] if condition == "corrupted_proprio" else live[key])
    assert actual["image"].dtype == np.uint8
    np.testing.assert_array_equal(live["image"], observation(2)["image"])
    fed = policy_observation(actual, "cpu")
    assert "to_target" not in fed and "reward" not in fed
    assert fed["image"].shape == (1, 8, 8, 3)
    assert fed["position"].shape == (1, 2)


def test_fake_target_changes_render_only():
    pytest.importorskip("dm_control")
    cfg = OmegaConf.create({"task": "dmc_reacher_hard", "action_repeat": 2, "size": [64, 64], "time_limit": 20})
    normal = make_evaluation_env(cfg, 123)
    fake = make_evaluation_env(cfg, 123, sample_target(987))
    try:
        obs_normal, obs_fake = normal.reset(), fake.reset()
        physics = fake.unwrapped._env.physics
        original_state = physics.get_state().copy()
        original_model_target = physics.named.model.geom_pos["target"].copy()
        reward = fake.unwrapped._env.task.get_reward(physics)
        assert not np.array_equal(obs_normal["image"], obs_fake["image"])
        np.testing.assert_array_equal(fake.unwrapped.render_true(), obs_normal["image"])
        for _ in range(3):
            fake.unwrapped.render()
        np.testing.assert_array_equal(physics.get_state(), original_state)
        np.testing.assert_array_equal(physics.named.model.geom_pos["target"], original_model_target)
        assert fake.unwrapped._env.task.get_reward(physics) == reward
        for _ in range(10):
            a, ra, da, _ = normal.step(np.array([0.1, -0.2], np.float32))
            b, rb, db, _ = fake.step(np.array([0.1, -0.2], np.float32))
            assert ra == rb and da == db
            for key in ("position", "velocity", "to_target"):
                np.testing.assert_array_equal(a[key], b[key])
            np.testing.assert_array_equal(normal.unwrapped._env.physics.get_state(), physics.get_state())
    finally:
        normal.close()
        fake.close()


def test_paired_summary_is_reproducible():
    rows = []
    for seed in range(10):
        rows += [
            {"condition": "baseline", "seed": seed, "return": seed * 10.0},
            {"condition": "black", "seed": seed, "return": seed * 10.0 - 5.0},
        ]
    result = summarize(rows, samples=100)
    assert result == summarize(rows, samples=100)
    assert result[0]["mean_paired_delta"] == 0.0
    assert result[1]["mean_paired_delta"] == -5.0
    assert result[1]["paired_delta_ci95"] == [-5.0, -5.0]


def test_evaluation_freezes_checkpoint_and_is_reproducible(tmp_path):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from dreamer import Dreamer
    from evaluate_ablations import load_agent, rollout

    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        config = compose(
            config_name="configs",
            overrides=[
                "device=cpu",
                "env.task=dmc_reacher_hard",
                "env.time_limit=8",
                "model.compile=False",
                "model.deter=64",
                "model.hidden=64",
                "model.units=64",
                "model.depth=4",
                "model.discrete=4",
            ],
        )
    config.env.encoder.mlp_keys = "^(position|velocity)$"
    saved_config = OmegaConf.to_container(config, resolve=True)
    env = make_evaluation_env(config.env, 0)
    try:
        model = Dreamer(config.model, env.observation_space, env.action_space).to("cpu")
    finally:
        env.close()
    path = tmp_path / "latest.pt"
    torch.save(
        {"config": saved_config, "agent": model.state_dict(), "trainer": {"step": 123}},
        path,
    )
    agent, loaded_config, step = load_agent(path, torch.device("cpu"))
    assert step == 123 and not agent.training
    assert not any(parameter.requires_grad for parameter in agent.parameters())
    before = {key: value.clone() for key, value in agent.state_dict().items()}
    reference, _, _, _ = rollout(agent, loaded_config.env, 200)
    _, donor, _, _ = rollout(agent, loaded_config.env, 201)
    repeat, _, _, _ = rollout(agent, loaded_config.env, 200)
    assert reference["return"] == repeat["return"]
    assert reference["initial_state"] == repeat["initial_state"]
    for condition in CONDITIONS:
        row, _, _, diagnostic = rollout(agent, loaded_config.env, 200, condition, donor, record=True)
        assert row["initial_state"] == reference["initial_state"]
        assert row["real_target"] == reference["real_target"]
        if condition == "corrupted_proprio":
            assert not np.array_equal(diagnostic["true_proprio"], diagnostic["input_proprio"])
    for key, value in agent.state_dict().items():
        assert torch.equal(value, before[key])
    assert not agent._optimizer.state
