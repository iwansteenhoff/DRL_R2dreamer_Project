import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from buffer import Buffer
from disk_replay import CheckpointMemmapStorage


def make_buffer(path, capacity=12, update_latents=False, backend="memmap"):
    return Buffer(
        SimpleNamespace(
            device="cpu",
            storage_device="cpu",
            backend=backend,
            directory=str(path),
            max_size=capacity,
            batch_size=1,
            batch_length=2,
            online=True,
            update_latents=update_latents,
        )
    )


def add(buffer, value):
    buffer.add_transition(
        TensorDict(
            {
                "image": torch.full((2, 8, 8, 3), value, dtype=torch.uint8),
                "episode": torch.tensor([0, 1]),
                "action": torch.zeros(2, 1),
                "stoch": torch.full((2, 1), float(value)),
                "deter": torch.full((2, 1), float(value)),
            },
            batch_size=[2],
        )
    )


def snapshot(buffer):
    state = buffer.state_dict()
    stream = io.BytesIO()
    torch.save(state, stream)
    buffer.checkpoint_committed(state)
    stream.seek(0)
    return torch.load(stream, weights_only=False)


def values(buffer):
    storage = buffer._buffer.storage
    return storage._storage[: storage.shape[0]].clone()


@pytest.mark.parametrize("initial,extra", [(3, 2), (5, 4), (6, 10)])
def test_restore_after_append_wrap_and_multiple_wraps(tmp_path, initial, extra):
    buffer = make_buffer(tmp_path)
    for i in range(initial):
        add(buffer, i)
    saved = snapshot(buffer)
    expected = values(buffer)
    cursor = buffer._buffer._writer._cursor
    for i in range(extra):
        add(buffer, 20 + i)
    restored = make_buffer(tmp_path)
    restored.load_state_dict(saved)
    assert restored.count() == initial * 2
    assert restored._buffer._writer._cursor == cursor
    assert (values(restored) == expected).all()
    assert not restored._online._pending
    data, _, _ = restored.sample(prefer_online=False)
    assert data["image"].shape == (1, 2, 8, 8, 3)
    # Repeating recovery is safe even after the journal has been removed.
    restored.load_state_dict(saved)
    assert (values(restored) == expected).all()


def test_undo_updates_to_existing_latents(tmp_path):
    buffer = make_buffer(tmp_path, update_latents=True)
    for i in range(4):
        add(buffer, i)
    saved = snapshot(buffer)
    expected = values(buffer)
    buffer.update(
        [torch.tensor([[0, 1]]), torch.tensor([[0, 1]])], torch.full((1, 2, 1), 99.0), torch.full((1, 2, 1), 98.0)
    )
    restored = make_buffer(tmp_path, update_latents=True)
    restored.load_state_dict(saved)
    assert (values(restored) == expected).all()


def test_failed_journal_write_prevents_overwrite(tmp_path):
    buffer = make_buffer(tmp_path)
    for i in range(6):
        add(buffer, i)
    snapshot(buffer)
    expected = values(buffer)
    with patch("disk_replay.torch.save", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            add(buffer, 55)
    assert (values(buffer) == expected).all()


def test_new_checkpoint_ignores_previous_transaction_journal(tmp_path):
    buffer = make_buffer(tmp_path)
    for i in range(6):
        add(buffer, i)
    old = snapshot(buffer)
    add(buffer, 70)
    candidate = buffer.state_dict()
    stream = io.BytesIO()
    torch.save(candidate, stream)
    stream.seek(0)
    candidate = torch.load(stream, weights_only=False)
    expected = values(buffer)
    # Simulate a crash after latest.pt replacement, before journal cleanup.
    restored = make_buffer(tmp_path)
    restored.load_state_dict(candidate)
    assert (values(restored) == expected).all()
    assert not (tmp_path / "undo" / old["replay"]["_storage"]["token"]).exists()


def test_legacy_conversion_only_copies_valid_entries(tmp_path):
    old = make_buffer(tmp_path / "unused", backend="memory")
    for i in range(3):
        add(old, i)
    expected = values(old)
    new = make_buffer(tmp_path / "disk")
    new.load_state_dict(old.state_dict())
    assert (values(new) == expected).all()
    saved = snapshot(new)
    restored = make_buffer(tmp_path / "disk")
    restored.load_state_dict(saved)
    assert (values(restored) == expected).all()


def test_checkpoint_excludes_replay_capacity_and_requires_data_files(tmp_path):
    buffer = make_buffer(tmp_path, capacity=10000)
    for i in range(3):
        add(buffer, i)
    saved = snapshot(buffer)
    stream = io.BytesIO()
    torch.save(saved, stream)
    assert stream.tell() < 100_000  # Capacity would require several MB of images.
    assert isinstance(buffer._buffer.storage, CheckpointMemmapStorage)
    image = tmp_path / "data" / "image.memmap"
    image.unlink()
    with pytest.raises(ValueError, match="Missing or truncated"):
        make_buffer(tmp_path, capacity=10000).load_state_dict(saved)


def test_interrupted_rollback_can_be_retried(tmp_path):
    buffer = make_buffer(tmp_path)
    for i in range(6):
        add(buffer, i)
    saved = snapshot(buffer)
    expected = values(buffer)
    for i in range(3):
        add(buffer, 50 + i)
    real_load = torch.load
    calls = 0

    def interrupted_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("interrupted recovery")
        return real_load(*args, **kwargs)

    with patch("disk_replay.torch.load", side_effect=interrupted_load):
        with pytest.raises(OSError, match="interrupted recovery"):
            make_buffer(tmp_path).load_state_dict(saved)
    restored = make_buffer(tmp_path)
    restored.load_state_dict(saved)
    assert (values(restored) == expected).all()


def test_failed_checkpoint_keeps_old_replay_recoverable(tmp_path):
    import checkpoint

    buffer = make_buffer(tmp_path / "replay")
    for i in range(6):
        add(buffer, i)
    model = torch.nn.Linear(1, 1)
    model.device = torch.device("cpu")
    model._optimizer = torch.optim.Adam(model.parameters())
    model._scheduler = torch.optim.lr_scheduler.StepLR(model._optimizer, 1)
    trainer = SimpleNamespace(
        logger=SimpleNamespace(_logdir=tmp_path),
        replay_buffer=buffer,
        started_at=0.0,
        step=12,
        state_dict=lambda: {"step": 12},
    )
    path = tmp_path / "latest.pt"
    checkpoint.save(path, {}, model, trainer)
    state = torch.load(path, weights_only=False)
    expected = values(buffer)
    for i in range(3):
        add(buffer, 90 + i)
    before = path.read_bytes()
    with patch("checkpoint.torch.save", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            checkpoint.save(path, {}, model, trainer)
    assert path.read_bytes() == before
    restored = make_buffer(tmp_path / "replay")
    restored.load_state_dict(state["replay"])
    assert (values(restored) == expected).all()
