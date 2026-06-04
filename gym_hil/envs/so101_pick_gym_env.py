#!/usr/bin/env python

from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
_SO101_HOME = np.asarray((0.0, -0.6, 0.9, -0.5, 0.0, 0.35), dtype=np.float32)
_SO101_ACTION_SCALE = np.asarray((0.05, 0.05, 0.05, 0.05, 0.08, 0.08), dtype=np.float32)
_SAMPLING_BOUNDS = np.asarray([[0.16, -0.08], [0.28, 0.08]])


class SO101PickCubeGymEnv(MujocoGymEnv):
    """Joint-position SO-101 pick-cube environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        random_block_position: bool = True,
        xml_path: Path | None = None,
    ):
        self.reward_type = reward_type
        self.render_mode = render_mode
        self.image_obs = image_obs
        self._random_block_position = random_block_position

        if xml_path is None:
            xml_path = Path(__file__).parent.parent / "assets" / "so101_pick_scene.xml"

        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self._joint_ids = np.asarray([self._model.joint(name).id for name in _SO101_JOINT_NAMES])
        self._qpos_ids = self._model.jnt_qposadr[self._joint_ids]
        self._qvel_ids = self._model.jnt_dofadr[self._joint_ids]
        self._ctrl_ids = np.asarray([self._model.actuator(name).id for name in _SO101_JOINT_NAMES])
        self._ctrl_range = self._model.actuator_ctrlrange[self._ctrl_ids]

        self._pinch_site_id = self._model.site("pinch").id
        self._block_z = self._model.geom("block").size[2]

        camera_id_1 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        camera_id_2 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "handcam_rgb")
        self.camera_id = (camera_id_1, camera_id_2)

        agent_dim = self.get_robot_state().shape[0]
        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)
        env_box = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        if self.image_obs:
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "front": spaces.Box(
                                0,
                                255,
                                (self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": spaces.Box(
                                0,
                                255,
                                (self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                    "agent_pos": agent_box,
                }
            )
        else:
            self.observation_space = spaces.Dict({"agent_pos": agent_box, "environment_state": env_box})

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(_SO101_JOINT_NAMES),), dtype=np.float32)
        self._viewer = mujoco.Renderer(self.model, height=render_spec.height, width=render_spec.width)
        self._viewer.render()

    def reset_robot(self):
        """Reset the SO-101 joints and position-servo targets."""
        self._data.qpos[self._qpos_ids] = _SO101_HOME
        self._data.qvel[self._qvel_ids] = 0.0
        self._data.ctrl[self._ctrl_ids] = _SO101_HOME
        mujoco.mj_forward(self._model, self._data)

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)
        self.reset_robot()

        if self._random_block_position:
            block_xy = self.random_state.uniform(*_SAMPLING_BOUNDS)
        else:
            block_xy = np.asarray([0.24, 0.0])
        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        mujoco.mj_forward(self._model, self._data)

        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.05
        return self._compute_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Apply a scaled joint-position delta action."""
        self.apply_action(action)
        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._is_success() if self.reward_type == "dense" else rew == 1.0
        return obs, rew, bool(success), False, {"succeed": bool(success)}

    def apply_action(self, action: np.ndarray):
        """Apply SO-101 joint-position delta commands."""
        action = np.asarray(action, dtype=np.float32)
        target = self._data.ctrl[self._ctrl_ids] + action * _SO101_ACTION_SCALE
        target = np.clip(target, self._ctrl_range[:, 0], self._ctrl_range[:, 1])
        self._data.ctrl[self._ctrl_ids] = target
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

    def get_robot_state(self):
        """Return joint positions, joint velocities, gripper command, and TCP position."""
        qpos = self._data.qpos[self._qpos_ids].astype(np.float32)
        qvel = self._data.qvel[self._qvel_ids].astype(np.float32)
        gripper_pose = np.asarray([self._data.ctrl[self._ctrl_ids[-1]]], dtype=np.float32)
        tcp_pos = self._data.sensor("so101/pinch_pos").data.astype(np.float32)
        return np.concatenate([qpos, qvel, gripper_pose, tcp_pos])

    def render(self):
        """Render frames from the front and wrist cameras."""
        rendered_frames = []
        for cam_id in self.camera_id:
            self._viewer.update_scene(self.data, camera=cam_id)
            rendered_frames.append(self._viewer.render())
        return rendered_frames

    def _compute_observation(self) -> dict:
        robot_state = self.get_robot_state().astype(np.float32)
        block_pos = self._data.sensor("block_pos").data.astype(np.float32)
        if self.image_obs:
            front_view, wrist_view = self.render()
            return {"pixels": {"front": front_view, "wrist": wrist_view}, "agent_pos": robot_state}
        return {"agent_pos": robot_state, "environment_state": block_pos}

    def _compute_reward(self) -> float:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("so101/pinch_pos").data
        dist = np.linalg.norm(block_pos - tcp_pos)
        if self.reward_type == "dense":
            r_close = np.exp(-20 * dist)
            r_lift = np.clip((block_pos[2] - self._z_init) / (self._z_success - self._z_init), 0.0, 1.0)
            return float(0.3 * r_close + 0.7 * r_lift)
        return float(block_pos[2] - self._z_init > 0.05)

    def _is_success(self) -> bool:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("so101/pinch_pos").data
        return bool(np.linalg.norm(block_pos - tcp_pos) < 0.04 and block_pos[2] - self._z_init > 0.05)
