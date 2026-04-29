"""Gymnasium wrappers that expose CV-predicted state to RL agents."""

from __future__ import annotations

import json
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

from .vision import PredictedState, StatePredictor
from .models.cv import build_cv_model

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
        self._prev_state: PredictedState | None = None

        self.observation_space = self._make_observation_space()
        self._validate_cv_outputs()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the base environment and return a CV-derived observation."""

        true_obs, info = self.env.reset(seed=seed, options=options)
        self._prev_state = None
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
        state = self.state_predictor.predict_state(self._render_rgb_frame())
        true_obs = np.asarray(true_obs, dtype=np.float32)
        theta = float(true_obs[4]) if state.theta is None else state.theta

        if self.obs_mode == "hybrid":
            obs = np.array(
                [
                    state.x,
                    state.y,
                    float(true_obs[2]),
                    float(true_obs[3]),
                    theta,
                    float(true_obs[5]),
                    float(true_obs[6]),
                    float(true_obs[7]),
                ],
                dtype=np.float32,
            )
        else:
            if state.theta is None:
                raise ValueError(
                    "obs_mode='cv-only' requires a CV model that predicts theta "
                    "or sin_theta/cos_theta. Use obs_mode='hybrid' for x_y models.",
                )
            vx, vy, angular_velocity = self._finite_difference_velocity(state)
            obs = np.array(
                [state.x, state.y, vx, vy, state.theta, angular_velocity, 0.0, 0.0],
                dtype=np.float32,
            )

        self._prev_state = state
        return obs, {"vision_pose": state.as_pose_array()}

    def _finite_difference_velocity(self, state: PredictedState) -> tuple[float, float, float]:
        if self._prev_state is None:
            return 0.0, 0.0, 0.0
        if self._prev_state.theta is None or state.theta is None:
            raise ValueError("Finite-difference angular velocity requires theta predictions.")

        vx = (state.x - self._prev_state.x) / self.diff_dt
        vy = (state.y - self._prev_state.y) / self.diff_dt
        dtheta = _wrap_angle(state.theta - self._prev_state.theta)
        return float(vx), float(vy), float(dtheta / self.diff_dt)

    def _validate_cv_outputs(self) -> None:
        output_columns = set(self.state_predictor.output_columns)
        if not {"x", "y"}.issubset(output_columns):
            raise ValueError(
                "CV output_columns must include x and y, "
                f"got {self.state_predictor.output_columns}.",
            )
        has_theta = "theta" in output_columns or {"sin_theta", "cos_theta"}.issubset(output_columns)
        if self.obs_mode == "cv-only" and not has_theta:
            raise ValueError(
                "obs_mode='cv-only' requires theta or sin_theta/cos_theta in CV output_columns. "
                "Use obs_mode='hybrid' for x_y models.",
            )


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def make_vision_lander_env(
    cv_weights: str | Path,
    cv_model_type: str = "resnet18",
    cv_metadata: str | Path | None = None,
    cv_output_columns: tuple[str, ...] | None = None,
    device: str = "auto",
    obs_mode: ObservationMode = "hybrid",
    seed: int | None = None,
    env_id: str = "LunarLander-v3",
) -> VisionStateLunarLanderWrapper:
    """Create a LunarLander env whose observations come from rendered frames."""

    output_columns = _resolve_cv_output_columns(cv_metadata, cv_output_columns)
    predictor = StatePredictor(
        cv_weights,
        device=device,
        model=build_cv_model(cv_model_type, out_dim=len(output_columns)),
        output_columns=output_columns,
    )
    env = gym.make(
        env_id,
        render_mode="rgb_array",
        enable_wind=False,
        continuous=False,
    )
    if seed is not None:
        env.action_space.seed(seed)
    return VisionStateLunarLanderWrapper(env, predictor, obs_mode=obs_mode)


def _resolve_cv_output_columns(
    cv_metadata: str | Path | None,
    cv_output_columns: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if cv_output_columns is not None:
        return tuple(cv_output_columns)
    if cv_metadata is None:
        return ("x", "y", "sin_theta", "cos_theta")

    metadata_path = Path(cv_metadata)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_output_columns = metadata.get("model_output_columns")
    if isinstance(model_output_columns, list) and all(isinstance(c, str) for c in model_output_columns):
        return tuple(model_output_columns)

    target_columns = metadata.get("target_columns")
    if not isinstance(target_columns, list) or not all(isinstance(c, str) for c in target_columns):
        raise ValueError(f"metadata target_columns must be a list of strings: {metadata_path}")

    angle_target = metadata.get("angle_target", "sincos")
    output_columns: list[str] = []
    for column in target_columns:
        if column == "theta" and angle_target == "sincos":
            output_columns.extend(["sin_theta", "cos_theta"])
        else:
            output_columns.append(column)
    return tuple(output_columns)
