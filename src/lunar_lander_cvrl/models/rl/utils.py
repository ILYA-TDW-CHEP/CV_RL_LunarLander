"""Utilities for Stable-Baselines3 RL training and evaluation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.base_class import BaseAlgorithm
except ImportError as exc:  # pragma: no cover - depends on optional package.
    raise ImportError(
        "stable-baselines3 is required for RL utilities. "
        "Install dependencies with: pip install -r requirements.txt",
    ) from exc

from lunar_lander_cvrl.models.cv import CV_MODEL_TYPES

ALGORITHMS = ("dqn", "ppo")


def validate_training_args(args: SimpleNamespace) -> None:
    """Validate config values used by the RL training entrypoint."""

    _validate_common_args(args.algorithm, args.cv_model_type, args.obs_mode)
    if args.policy != "MlpPolicy":
        raise ValueError("Only rl.policy=MlpPolicy is supported by this training script.")
    if args.timesteps <= 0:
        raise ValueError("rl.timesteps must be positive.")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("rl.gamma must be in (0, 1].")
    if args.algorithm == "dqn":
        _validate_dqn_args(args)
    if args.algorithm == "ppo":
        _validate_ppo_args(args)
    if args.visualize and args.vis_freq <= 0:
        raise ValueError("visualization.freq must be positive.")
    if args.visualize and args.vis_max_steps < 0:
        raise ValueError("visualization.max_steps cannot be negative.")
    if args.visualize and args.vis_fps <= 0:
        raise ValueError("visualization.fps must be positive.")
    if args.checkpoint_freq < 0:
        raise ValueError("checkpoint.freq cannot be negative.")


def validate_evaluation_args(args: SimpleNamespace) -> None:
    """Validate config values used by the RL evaluation entrypoint."""

    _validate_common_args(args.algorithm, args.cv_model_type, args.obs_mode)
    if args.episodes <= 0:
        raise ValueError("evaluation.episodes must be positive.")


def get_algorithm_class(algorithm: str) -> type[BaseAlgorithm]:
    """Return the SB3 class for a supported algorithm name."""

    if algorithm == "dqn":
        return DQN
    if algorithm == "ppo":
        return PPO
    raise ValueError(f"Unsupported RL algorithm {algorithm!r}.")


def create_sb3_model(args: SimpleNamespace, env: Any) -> BaseAlgorithm:
    """Create a new SB3 model from the Hydra-derived training args."""

    model_cls = get_algorithm_class(args.algorithm)
    common_kwargs = {
        "verbose": 1,
        "seed": args.seed,
        "device": args.device,
    }

    if args.algorithm == "dqn":
        return model_cls(
            args.policy,
            env,
            learning_rate=args.dqn_learning_rate,
            gamma=args.gamma,
            buffer_size=args.dqn_buffer_size,
            learning_starts=args.dqn_learning_starts,
            batch_size=args.dqn_batch_size,
            train_freq=args.dqn_train_freq,
            gradient_steps=args.dqn_gradient_steps,
            target_update_interval=args.dqn_target_update_interval,
            exploration_fraction=args.dqn_exploration_fraction,
            exploration_final_eps=args.dqn_exploration_final_eps,
            **common_kwargs,
        )

    return model_cls(
        args.policy,
        env,
        learning_rate=args.ppo_learning_rate,
        gamma=args.gamma,
        n_steps=args.ppo_n_steps,
        batch_size=args.ppo_batch_size,
        n_epochs=args.ppo_n_epochs,
        gae_lambda=args.ppo_gae_lambda,
        clip_range=args.ppo_clip_range,
        ent_coef=args.ppo_ent_coef,
        vf_coef=args.ppo_vf_coef,
        **common_kwargs,
    )


def load_sb3_model(args: SimpleNamespace, env: Any, load_path: Path) -> BaseAlgorithm:
    """Load a saved SB3 model for the configured algorithm."""

    model_cls = get_algorithm_class(args.algorithm)
    return model_cls.load(load_path, env=env, device=args.device)


def select_load_path(save_path: Path, load_path: Path | None, resume: bool) -> Path | None:
    """Choose which model checkpoint should be loaded before training."""

    if resume and save_path.exists():
        return save_path
    if load_path is not None:
        if not load_path.exists():
            raise FileNotFoundError(f"Requested load.path does not exist: {load_path}")
        return load_path
    if resume:
        print(f"load.resume is true, but {save_path} does not exist yet. Starting from scratch.")
    return None


def resolve_replay_buffer_path(save_path: Path, replay_buffer_path: str | Path | None) -> Path:
    """Return the DQN replay-buffer path for a training run."""

    return Path(replay_buffer_path) if replay_buffer_path else save_path.with_suffix(".replay_buffer.pkl")


def load_replay_buffer_if_available(model: BaseAlgorithm, replay_buffer_path: Path) -> None:
    """Load a DQN replay buffer when it exists."""

    if replay_buffer_path.exists():
        model.load_replay_buffer(replay_buffer_path)
        print(f"Loaded replay buffer from {replay_buffer_path}")
    else:
        print(f"Replay buffer not found at {replay_buffer_path}; continuing without it.")


def save_model_artifacts(
    model: BaseAlgorithm,
    algorithm: str,
    save_path: Path,
    replay_buffer_path: Path,
    save_replay_buffer: bool,
) -> None:
    """Save the trained SB3 model and optional DQN replay buffer."""

    model.save(save_path)
    if algorithm == "dqn" and save_replay_buffer:
        replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_replay_buffer(replay_buffer_path)
        print(f"Saved replay buffer to {replay_buffer_path}")
    print(f"Saved RL model to {save_path}")


def _validate_common_args(algorithm: str, cv_model_type: str, obs_mode: str) -> None:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}.")
    if cv_model_type not in CV_MODEL_TYPES:
        raise ValueError(f"cv.model_type must be one of {CV_MODEL_TYPES}, got {cv_model_type!r}.")
    if obs_mode not in ("hybrid", "cv-only"):
        raise ValueError("env.obs_mode must be either 'hybrid' or 'cv-only'.")


def _validate_dqn_args(args: SimpleNamespace) -> None:
    if args.dqn_buffer_size <= 0:
        raise ValueError("rl.dqn.buffer_size must be positive.")
    if args.dqn_learning_starts < 0:
        raise ValueError("rl.dqn.learning_starts cannot be negative.")
    if args.dqn_batch_size <= 1:
        raise ValueError("rl.dqn.batch_size must be greater than 1.")
    if args.dqn_train_freq <= 0:
        raise ValueError("rl.dqn.train_freq must be positive.")
    if args.dqn_gradient_steps < -1 or args.dqn_gradient_steps == 0:
        raise ValueError("rl.dqn.gradient_steps must be -1 or a positive integer.")
    if args.dqn_target_update_interval <= 0:
        raise ValueError("rl.dqn.target_update_interval must be positive.")
    if not 0.0 <= args.dqn_exploration_final_eps <= 1.0:
        raise ValueError("rl.dqn.exploration_final_eps must be in [0, 1].")
    if not 0.0 < args.dqn_exploration_fraction <= 1.0:
        raise ValueError("rl.dqn.exploration_fraction must be in (0, 1].")


def _validate_ppo_args(args: SimpleNamespace) -> None:
    if args.ppo_n_steps <= 1:
        raise ValueError("rl.ppo.n_steps must be greater than 1.")
    if args.ppo_batch_size <= 1:
        raise ValueError("rl.ppo.batch_size must be greater than 1.")
    if args.ppo_batch_size > args.ppo_n_steps:
        raise ValueError("rl.ppo.batch_size should be <= rl.ppo.n_steps for a single env.")
    if args.ppo_n_epochs <= 0:
        raise ValueError("rl.ppo.n_epochs must be positive.")
    if not 0.0 < args.ppo_gae_lambda <= 1.0:
        raise ValueError("rl.ppo.gae_lambda must be in (0, 1].")
    if args.ppo_ent_coef < 0.0:
        raise ValueError("rl.ppo.ent_coef cannot be negative.")
    if args.ppo_vf_coef < 0.0:
        raise ValueError("rl.ppo.vf_coef cannot be negative.")
