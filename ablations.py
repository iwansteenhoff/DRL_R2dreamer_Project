"""Reacher observation ablations that never change the task's physics or rewards."""

import numpy as np
import torch
from tensordict import TensorDict

from envs.dmc import DeepMindControl
from envs.wrappers import Dtype, NormalizeActions, TimeLimit

CONDITIONS = ("baseline", "wrong_target", "frozen", "unrelated", "black", "corrupted_proprio")
PROPRIO_KEYS = ("position", "velocity")
# every state key Reacher returns, to_target is only fed to models trained with it
STATE_KEYS = ("position", "velocity", "to_target")


def sample_target(seed):
    # Same angle/radius distribution as dm_control.suite.reacher.initialize_episode.
    rng = np.random.RandomState(seed)
    angle = rng.uniform(0, 2 * np.pi)
    radius = rng.uniform(0.05, 0.20)
    return radius * np.array([np.sin(angle), np.cos(angle)])


class ReacherEvaluationEnv(DeepMindControl):
    def __init__(self, name, action_repeat, size, seed, fake_target=None):
        if name not in ("reacher_easy", "reacher_hard"):
            raise ValueError("These ablations support standard DMC Reacher easy/hard only.")
        self.fake_target = None if fake_target is None else np.asarray(fake_target).copy()
        super().__init__(name, action_repeat=action_repeat, size=size, seed=seed)
        self.target_id = self._env.physics.model.name2id("target", "geom")

    def _move_rendered_target(self, physics, scene):
        import mujoco

        found = False
        for geom in scene.geoms[: scene.ngeom]:
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == self.target_id:
                geom.pos[:2] = self.fake_target
                found = True
        if not found:
            raise RuntimeError("Target geometry not found in rendered scene.")

    def render(self, *args, **kwargs):
        callback = self._move_rendered_target if self.fake_target is not None else None
        return self._env.physics.render(*self._size, camera_id=self._camera, scene_callback=callback).copy()

    def render_true(self):
        return self._env.physics.render(*self._size, camera_id=self._camera).copy()

    def geometry(self):
        physics = self._env.physics
        return {
            "target": physics.named.data.geom_xpos["target", :2].copy(),
            "endpoint": physics.named.data.geom_xpos["finger", :2].copy(),
        }

    def close(self):
        self._env.physics.free()


def make_evaluation_env(config, seed, fake_target=None, time_limit=None):
    raw = ReacherEvaluationEnv(
        config.task.removeprefix("dmc_"), int(config.action_repeat), tuple(config.size), seed, fake_target
    )
    limit = int(config.time_limit if time_limit is None else time_limit)
    if limit <= 0 or limit % int(config.action_repeat):
        raw.close()
        raise ValueError("Time limit must be positive and divisible by action_repeat.")
    return Dtype(TimeLimit(NormalizeActions(raw), limit // int(config.action_repeat)))


class ObservationAblation:
    def __init__(self, condition, initial_image, donor=None, state_keys=PROPRIO_KEYS):
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        if condition in ("unrelated", "corrupted_proprio") and not donor:
            raise ValueError("This condition requires a donor episode.")
        self.condition = condition
        self.initial_image = initial_image.copy()
        self.donor = donor
        # corrupted_proprio replaces every state input the model uses
        self.state_keys = tuple(state_keys)

    def apply(self, observation, step):
        result = dict(observation)
        if self.condition == "frozen":
            result["image"] = self.initial_image.copy()
        elif self.condition == "black":
            result["image"] = np.zeros_like(observation["image"])
        elif self.condition in ("unrelated", "corrupted_proprio"):
            if step >= len(self.donor):
                raise ValueError("Donor episode is shorter than the evaluated episode.")
            keys = ("image",) if self.condition == "unrelated" else self.state_keys
            for key in keys:
                result[key] = self.donor[step][key].copy()
        return result


def policy_observation(observation, device, state_keys=PROPRIO_KEYS):
    # Explicit whitelist: only the state keys the model was trained on reach the policy,
    # rewards and diagnostics never do.
    keys = ("image", *state_keys, "is_first", "is_last", "is_terminal")
    return TensorDict(
        {key: torch.as_tensor(np.array(observation[key], copy=True), device=device).unsqueeze(0) for key in keys},
        batch_size=[1],
        device=device,
    )
