from types import SimpleNamespace

import pytest
import torch

from dreamer import Dreamer


def fake_agent(probability, image_dim=3):
    # drop_modalities only needs the encoder layout and the dropout chance
    encoder = SimpleNamespace(
        cnn_shapes={"image": (64, 64, 3)},
        mlp_shapes={"position": (2,)},
        encoders=[SimpleNamespace(out_dim=image_dim)],
    )
    return SimpleNamespace(encoder=encoder, modality_dropout=probability)


def test_dropout_hides_one_modality_per_sequence():
    torch.manual_seed(0)
    embed = torch.ones(1000, 4, 5)
    out = Dreamer.drop_modalities(fake_agent(0.5), embed)
    image_kept = out[:, :, :3].sum(dim=(1, 2)) > 0
    state_kept = out[:, :, 3:].sum(dim=(1, 2)) > 0
    # never both hidden, and a hidden part is zero for every time step of the sequence
    assert (image_kept | state_kept).all()
    assert torch.all((out[:, :, :3] == 0).all(dim=(1, 2)) == ~image_kept)
    assert torch.all((out[:, :, 3:] == 0).all(dim=(1, 2)) == ~state_kept)
    # about a quarter of the sequences lose each modality
    assert 0.2 < (~image_kept).float().mean() < 0.3
    assert 0.2 < (~state_kept).float().mean() < 0.3


def test_dropout_with_probability_one_always_hides_something():
    out = Dreamer.drop_modalities(fake_agent(1.0), torch.ones(200, 2, 5))
    hidden = (out[:, :, :3] == 0).all(dim=(1, 2)) | (out[:, :, 3:] == 0).all(dim=(1, 2))
    assert hidden.all()


def test_dropout_needs_both_modalities():
    agent = fake_agent(0.5)
    agent.encoder.mlp_shapes = {}
    with pytest.raises(ValueError):
        Dreamer.drop_modalities(agent, torch.ones(2, 2, 5))
