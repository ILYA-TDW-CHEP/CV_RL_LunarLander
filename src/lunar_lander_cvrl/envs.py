"""Gymnasium wrappers that expose CV-predicted state to RL agents."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover - exercised only without optional deps.
    raise ImportError(
        "Gymnasium is required for LunarLander environments. "
        "Install project dependencies with: pip install -r requirements.txt",
    ) from exc

from .vision import PredictedPose, StatePredictor

ObservationMode = Literal["hybrid", "cv-only"]


class VisionStateLunarLanderWrapper(gym.Wrapper):
    """Replace LunarLander observations with state estimated from rendered frames.

    ``hybrid`` mode uses CV for ``x``, ``y`` and ``theta`` while preserving the
    true velocity/contact components from Gymnasium. ``cv-only`` mode estimates
    velocities by finite differences and sets leg contacts to zero.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env: gym.Env,
        state_predictor: StatePredictor,
        obs_mode: ObservationMode = "hybrid",
        diff_dt: float = 1.0,
    ) -> None:
        super().__init__(env)
        if obs_mode not in ("hybrid", "cv-only"):
            raise ValueError("obs_mode must be either 'hybrid' or 'cv-only'.")
        if diff_dt <= 0:
            raise ValueError("diff_dt must be positive.")

        self.state_predictor = state_predictor
        self.obs_mode = obs_mode
        self.diff_dt = float(diff_dt)
        self._prev_pose: PredictedPose | None = None

        self.observation_space = self._make_observation_space()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the base environment and return a CV-derived observation."""

        true_obs, info = self.env.reset(seed=seed, options=options)
        self._prev_pose = None
        obs, vision_info = self._build_observation(true_obs)
        info = dict(info)
        info.update(vision_info)
        return obs, info

    def step(self, action):
        """Step the base environment and transform the next rendered frame."""

        true_obs, reward, terminated, truncated, info = self.env.step(action)
        obs, vision_info = self._build_observation(true_obs)
        info = dict(info)
        info.update(vision_info)
        return obs, reward, terminated, truncated, info

    def _make_observation_space(self) -> gym.spaces.Box:
        if isinstance(self.env.observation_space, gym.spaces.Box):
            low = np.asarray(self.env.observation_space.low, dtype=np.float32).copy()
            high = np.asarray(self.env.observation_space.high, dtype=np.float32).copy()
            if low.shape == (8,) and high.shape == (8,):
                low[4] = -np.pi
                high[4] = np.pi
                return gym.spaces.Box(low=low, high=high, dtype=np.float32)

        return gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.pi, -np.inf, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.pi, np.inf, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _render_rgb_frame(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError(
                "Environment render() returned None. Create LunarLander with "
                "render_mode='rgb_array' before applying VisionStateLunarLanderWrapper.",
            )
        return np.asarray(frame)

    def _build_observation(self, true_obs) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        pose = self.state_predictor.predict_pose(self._render_rgb_frame())
        true_obs = np.asarray(true_obs, dtype=np.float32)

        if self.obs_mode == "hybrid":
            obs = np.array(
                [
                    pose.x,
                    pose.y,
                    float(true_obs[2]),
                    float(true_obs[3]),
                    pose.theta,
                    float(true_obs[5]),
                    float(true_obs[6]),
                    float(true_obs[7]),
                ],
                dtype=np.float32,
            )
        else:
            vx, vy, angular_velocity = self._finite_difference_velocity(pose)
            obs = np.array(
                [pose.x, pose.y, vx, vy, pose.theta, angular_velocity, 0.0, 0.0],
                dtype=np.float32,
            )

        self._prev_pose = pose
        return obs, {"vision_pose": pose.as_pose_array()}

    def _finite_difference_velocity(self, pose: PredictedPose) -> tuple[float, float, float]:
        if self._prev_pose is None:
            return 0.0, 0.0, 0.0

        vx = (pose.x - self._prev_pose.x) / self.diff_dt
        vy = (pose.y - self._prev_pose.y) / self.diff_dt
        dtheta = _wrap_angle(pose.theta - self._prev_pose.theta)
        return float(vx), float(vy), float(dtheta / self.diff_dt)


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def make_vision_lander_env(
    cv_weights: str | Path,
    device: str = "auto",
    obs_mode: ObservationMode = "hybrid",
    seed: int | None = None,
    env_id: str = "LunarLander-v3",
) -> VisionStateLunarLanderWrapper:
    """Create a LunarLander env whose observations come from rendered frames."""

    predictor = StatePredictor(cv_weights, device=device)
    env = gym.make(
        env_id,
        render_mode="rgb_array",
        enable_wind=False,
        continuous=False,
    )
    if seed is not None:
        env.action_space.seed(seed)
    return VisionStateLunarLanderWrapper(env, predictor, obs_mode=obs_mode)
