"""Memory-mapped replay with rollback to the last durable model checkpoint."""

import os
import pathlib
import shutil
import uuid

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyMemmapStorage


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class CheckpointMemmapStorage(LazyMemmapStorage):
    """Keep images on disk; journal committed rows before in-place mutation.

    Only the training process may write this directory. Journals are scoped to a
    checkpoint token: an interrupted commit cannot apply an older undo journal
    to a newer, successfully saved model checkpoint.
    """

    def __init__(self, max_size, directory):
        self.directory = pathlib.Path(directory).expanduser().resolve()
        self.data_directory = self.directory / "data"
        self.token = None
        self.committed_rows = 0
        self.journaled = set()
        self.identity = None
        super().__init__(max_size=max_size, scratch_dir=str(self.data_directory), device="cpu", ndim=2)

    def _init(self, data):
        if self.data_directory.exists():
            raise FileExistsError(
                f"Replay already exists at {self.directory}; resume its checkpoint or use a new logdir."
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = uuid.uuid4().hex
        identity_path = self.directory / "identity"
        with identity_path.open("w") as stream:
            stream.write(self.identity)
            stream.flush()
            os.fsync(stream.fileno())
        super()._init(data)
        self.flush()

    def flush(self):
        """Make dirty mmap pages and metadata durable, without serializing them."""
        if not self.initialized:
            return
        for path in self.data_directory.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        for path in sorted(self.data_directory.rglob("*"), reverse=True):
            if path.is_dir():
                sync_directory(path)
        sync_directory(self.data_directory)
        sync_directory(self.directory)

    def _protect(self, cursor):
        if self.token is None or not self.initialized:
            return
        rows = cursor[0] if isinstance(cursor, tuple) else cursor
        if isinstance(rows, slice):
            rows = range(*rows.indices(self._storage.shape[0]))
        elif isinstance(rows, int):
            rows = [rows]
        elif isinstance(rows, torch.Tensor):
            rows = rows.detach().cpu().flatten().tolist()
        for row in set(int(row) % self._storage.shape[0] for row in rows):
            if row >= self.committed_rows or row in self.journaled:
                continue
            journal = self.directory / "undo" / self.token
            journal.mkdir(parents=True, exist_ok=True)
            sync_directory(self.directory)
            sync_directory(journal.parent)
            destination = journal / f"{row}.pt"
            temporary = destination.with_suffix(".tmp")
            # Persist the entire original row before any of its fields change.
            with temporary.open("wb") as stream:
                torch.save(self._storage[row].clone().to("cpu"), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            sync_directory(journal)
            self.journaled.add(row)

    def set(self, cursor, data, *, set_cursor=True):
        self._protect(cursor)
        super().set(cursor, data, set_cursor=set_cursor)

    def state_dict(self):
        self.flush()
        return {
            "disk_format": 1,
            "initialized": self.initialized,
            "_len": self._len,
            "max_size": self.max_size,
            "identity": self.identity,
            "token": uuid.uuid4().hex,
            "files": {
                str(path.relative_to(self.data_directory)): path.stat().st_size
                for path in self.data_directory.rglob("*")
                if path.is_file()
            },
        }

    def commit(self, state):
        """Called only after latest.pt has been atomically replaced and fsynced."""
        self.token = state["token"]
        self.committed_rows = self._len // self._storage.shape[1] if self.initialized else 0
        self.journaled.clear()
        undo = self.directory / "undo"
        if undo.exists():
            shutil.rmtree(undo)
            sync_directory(self.directory)

    def load_state_dict(self, state):
        if "disk_format" not in state:
            self._import_legacy(state)
            return
        if state["disk_format"] != 1:
            raise ValueError("Unsupported disk replay format.")
        if not state["initialized"]:
            # No committed entries exist; discard any data from after that snapshot.
            if self.data_directory.exists():
                shutil.rmtree(self.data_directory)
            self.initialized = False
            self._len = 0
            self.identity = None
            self.token = None
            return
        identity = self.directory / "identity"
        if not identity.exists() or identity.read_text() != state["identity"]:
            raise ValueError(f"Missing or mismatched replay directory: {self.directory}. Keep it with latest.pt.")
        for relative, size in state["files"].items():
            path = self.data_directory / relative
            if not path.is_file() or path.stat().st_size != size:
                raise ValueError(f"Missing or truncated replay file: {path}")
        self._storage = TensorDict.load_memmap(self.data_directory)
        self.initialized = True
        self.max_size = state["max_size"]
        self._len = state["_len"]
        self.identity = state["identity"]
        journal = self.directory / "undo" / state["token"]
        # Recovery is idempotent: keep journals until restored data is flushed.
        if journal.exists():
            for path in sorted(journal.glob("*.pt")):
                self._storage[int(path.stem)] = torch.load(path, map_location="cpu", weights_only=False)
            self.flush()
        self.commit(state)

    def _import_legacy(self, state):
        """One-time conversion of a full replay checkpoint, copying valid rows only."""
        if self.data_directory.exists():
            # An interrupted conversion is safe to retry: latest.pt still owns all data.
            shutil.rmtree(self.data_directory)
        if not state["initialized"]:
            self._len = 0
            return
        print("Converting full replay checkpoint to memory-mapped storage (populated entries only).")
        source = TensorDict().load_state_dict(state["_storage"], strict=False)
        self._init(source[0])
        rows = state["_len"] // source.shape[1]
        for start in range(0, rows, 128):
            self._storage[start : min(start + 128, rows)] = source[start : min(start + 128, rows)]
        self._len = state["_len"]
        self.flush()
