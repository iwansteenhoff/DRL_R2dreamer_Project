import atexit
import fcntl
import pathlib
import signal
import sys
import warnings

import hydra
import torch

import checkpoint
import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    # In-place replay has one writer. Never allow two resumed runs to share it.
    run_lock = (logdir / ".train.lock").open("a")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another training process is using {logdir}.") from error

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    checkpoint_path = logdir / "latest.pt"
    restored = checkpoint.load(checkpoint_path, config) if config.resume else None
    if not config.resume and (checkpoint_path.exists() or (logdir / "metrics.jsonl").exists()):
        raise ValueError("Log directory already contains a run; use resume=True or a new logdir.")
    config_state = checkpoint.experiment_config(config)
    if restored:
        # Remove records from work performed after the last successful checkpoint.
        metrics_path = logdir / "metrics.jsonl"
        if metrics_path.exists():
            with metrics_path.open("r+b") as stream:
                stream.truncate(restored["metrics_bytes"])
    logger = tools.Logger(logdir, purge_step=restored["trainer"]["step"] + 1 if restored else None)
    # save config
    logger.log_hydra_config(config)

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)

    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)
    if restored:
        checkpoint.restore(restored, agent, replay_buffer)
        print("Resuming at environment step", restored["trainer"]["step"])
        # Release old full-checkpoint mappings after migration. Otherwise the
        # replaced 11 GB file would remain allocated until this process exits.
        restored = {"trainer": restored["trainer"]}

    # Handle interruptions at a complete loop boundary, never mid optimizer update.
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
        print("Stop requested; saving after the current training iteration.")

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        policy_trainer.begin(
            agent,
            checkpoint=lambda agent, trainer: checkpoint.save(checkpoint_path, config_state, agent, trainer),
            restored=restored["trainer"] if restored else None,
            checkpoint_every=int(config.checkpoint.every),
            stop_requested=lambda: stopping,
        )
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        logger._writer.close()


if __name__ == "__main__":
    main()
