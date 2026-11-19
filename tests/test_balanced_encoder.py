import copy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from checkpoint import experiment_config
from networks import MultiEncoder


def encoder_config():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        config = compose(config_name="configs", overrides=["device=cpu", "model.depth=4", "model.units=64"])
    config.env.encoder.mlp_keys = "^(position|velocity)$"
    return OmegaConf.create(OmegaConf.to_container(config.model.encoder, resolve=True))


@pytest.mark.parametrize("full_state", [False, True])
def test_equal_width_and_gradients(full_state):
    config = encoder_config()
    config.fusion = "balanced"
    config.mlp_keys = "^(position|velocity|to_target)$" if full_state else "^(position|velocity)$"
    shapes = {"image": (64, 64, 3), "position": (2,), "velocity": (2,), "to_target": (2,)}
    encoder = MultiEncoder(config, shapes)
    observations = {key: torch.randn(2, 3, *shape) for key, shape in shapes.items()}
    features = [
        projection(branch(selector(observations)))
        for branch, selector, projection in zip(encoder.encoders, encoder.selectors, encoder.projections)
    ]
    assert [feature.shape for feature in features] == [torch.Size([2, 3, 640]), torch.Size([2, 3, 640])]
    output = encoder(observations)
    assert encoder.out_dim == 1280 and output.shape == (2, 3, 1280)
    torch.testing.assert_close(output, torch.cat(features, dim=-1))
    output.square().mean().backward()
    for branch, projection in zip(encoder.encoders, encoder.projections):
        assert projection.weight.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters())
    assert ("to_target" in encoder.mlp_shapes) == full_state
    restored = MultiEncoder(config, shapes)
    restored.load_state_dict(encoder.state_dict(), strict=True)
    torch.testing.assert_close(output, restored(observations))


def test_original_encoder_is_unchanged_with_new_defaults():
    config = encoder_config()
    legacy = copy.deepcopy(config)
    del legacy.fusion
    del legacy.fusion_dim
    shapes = {"image": (64, 64, 3), "position": (2,), "velocity": (2,)}
    torch.manual_seed(7)
    old = MultiEncoder(legacy, shapes)
    torch.manual_seed(7)
    new = MultiEncoder(config, shapes)
    assert old.out_dim == new.out_dim == 320
    assert set(old.state_dict()) == set(new.state_dict())
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key])


@pytest.mark.parametrize("dimension", [0, -2, 1279, 1280.5, True])
def test_reject_invalid_balanced_dimensions(dimension):
    config = encoder_config()
    config.fusion = "balanced"
    config.fusion_dim = dimension
    with pytest.raises(ValueError, match="positive even integer"):
        MultiEncoder(config, {"image": (64, 64, 3), "position": (2,)})


def test_balanced_requires_both_modalities():
    config = encoder_config()
    config.fusion = "balanced"
    with pytest.raises(ValueError, match="both image and vector"):
        MultiEncoder(config, {"image": (64, 64, 3)})


def test_checkpoint_compatibility_only_ignores_inactive_fusion_options():
    old = OmegaConf.create({"model": {"encoder": {}}, "env": {}, "trainer": {}})
    new = copy.deepcopy(old)
    new.model.encoder.fusion = "concat"
    new.model.encoder.fusion_dim = 1280
    assert experiment_config(old) == experiment_config(new)
    new.model.encoder.fusion = "balanced"
    assert experiment_config(old) != experiment_config(new)
    changed = copy.deepcopy(new)
    changed.model.encoder.fusion_dim = 1024
    assert experiment_config(changed) != experiment_config(new)


@pytest.mark.parametrize("fusion", ["balanced", "balanced_direct"])
def test_balanced_model_can_train_one_update(fusion):
    from buffer import Buffer
    from dreamer import Dreamer
    from tensordict import TensorDict

    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        config = compose(
            config_name="configs",
            overrides=[
                "device=cpu",
                "model.compile=False",
                "model.deter=64",
                "model.hidden=64",
                "model.units=64",
                "model.depth=4",
                "model.discrete=4",
                "batch_size=2",
                "batch_length=4",
                "buffer.backend=memory",
                "buffer.max_size=64",
                "model.encoder.fusion=balanced",
            ],
        )
    config.env.encoder.mlp_keys = "^(position|velocity)$"
    config.model.encoder.fusion = fusion
    config.model.imag_horizon = 3
    import gymnasium as gym
    import numpy as np

    obs_space = gym.spaces.Dict({
        "image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
        "position": gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
        "velocity": gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
    })
    act_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    model = Dreamer(config.model, obs_space, act_space).to("cpu")
    assert model.embed_size == 1280
    replay = Buffer(config.buffer)
    initial = model.get_initial_state(2)
    for step in range(8):
        replay.add_transition(
            TensorDict(
                {
                    "image": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8),
                    "position": torch.randn(2, 2),
                    "velocity": torch.randn(2, 2),
                    "action": torch.zeros(2, 2),
                    "reward": torch.zeros(2, 1),
                    "is_first": torch.full((2, 1), step == 0, dtype=torch.bool),
                    "is_last": torch.zeros(2, 1, dtype=torch.bool),
                    "is_terminal": torch.zeros(2, 1, dtype=torch.bool),
                    "episode": torch.arange(2),
                    "stoch": initial["stoch"],
                    "deter": initial["deter"],
                },
                batch_size=[2],
            )
        )
    parameter = next(model.encoder.encoders[1].parameters())
    before = parameter.detach().clone()
    model.update(replay)
    assert model._scheduler.last_epoch == 1
    assert model._nf_skipped == 0
    assert not torch.equal(before, parameter)


@pytest.mark.parametrize("full_state", [False, True])
def test_direct_outputs_and_parameter_count(full_state):
    config = encoder_config()
    config.cnn.depth = 16
    config.mlp.units = 256
    config.mlp_keys = "^(position|velocity|to_target)$" if full_state else "^(position|velocity)$"
    shapes = {"image": (64, 64, 3), "position": (2,), "velocity": (2,), "to_target": (2,)}
    original = MultiEncoder(config, shapes)
    config.fusion = "balanced_direct"
    direct = MultiEncoder(config, shapes)
    assert not len(direct.projections)
    assert direct.encoders[0].depths == (32, 48, 64, 40)
    assert [e.out_dim for e in direct.encoders] == [640, 640]
    mlp = direct.encoders[1].layers
    assert [getattr(mlp, f"mlp_encoder_linear{i}").out_features for i in range(3)] == [256, 256, 640]
    assert sum(p.numel() for p in direct.parameters()) - sum(p.numel() for p in original.parameters()) == 60624
    obs = {key: torch.randn(2, *shape) for key, shape in shapes.items()}
    output = direct(obs)
    assert output.shape == (2, 1280)
    output.square().mean().backward()
    assert all(p.grad is not None for p in direct.parameters())
    clone = MultiEncoder(config, shapes)
    clone.load_state_dict(direct.state_dict())
    torch.testing.assert_close(output, clone(obs))


def test_direct_rejects_incompatible_spatial_dimension():
    config = encoder_config()
    config.fusion = "balanced_direct"
    config.fusion_dim = 1282
    with pytest.raises(ValueError, match="spatial area"):
        MultiEncoder(config, {"image": (64, 64, 3), "position": (2,)})
