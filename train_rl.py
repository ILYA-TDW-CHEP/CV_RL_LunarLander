"""Train a DQN agent on CV-derived LunarLander observations."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

try:
    import hydra
except ImportError:  # pragma: no cover - exercised only without optional deps.
    hydra = None

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:  # pragma: no cover - depends on optional package.
    raise SystemExit(
        "stable-baselines3 is required for RL training. "
        "Install dependencies with: pip install -r requirements.txt",
    ) from exc

from lunar_lander_cvrl.envs import make_vision_lander_env
from lunar_lander_cvrl.models.cv import CV_MODEL_TYPES
from lunar_lander_cvrl.visualization import TrainingVisualizationCallback


def run_training(args: SimpleNamespace) -> None:
    _validate_args(args)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    replay_buffer_path = (
        Path(args.replay_buffer_path)
        if args.replay_buffer_path
        else save_path.with_suffix(".replay_buffer.pkl")
    )

    env = Monitor(
        make_vision_lander_env(
            cv_weights=args.cv_weights,
            cv_model_type=args.cv_model_type,
            cv_metadata=args.cv_metadata,
            device=args.device,
            obs_mode=args.obs_mode,
            seed=args.seed,
        ),
    )

    callbacks = []
    if args.visualize:
        callbacks.append(
            TrainingVisualizationCallback(
                eval_env=make_vision_lander_env(
                    cv_weights=args.cv_weights,
                    cv_model_type=args.cv_model_type,
                    cv_metadata=args.cv_metadata,
                    device=args.device,
                    obs_mode=args.obs_mode,
                    seed=args.seed + 10_000,
                ),
                output_dir=args.vis_dir,
                eval_freq=args.vis_freq,
                max_episode_steps=args.vis_max_steps or None,
                fps=args.vis_fps,
                seed=args.seed + 20_000,
                verbose=1,
            ),
        )

    if args.checkpoint_freq > 0:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix=save_path.stem,
                save_replay_buffer=not args.no_save_replay_buffer,
                save_vecnormalize=True,
            ),
        )

    callback = CallbackList(callbacks) if callbacks else None

    load_path = _select_load_path(
        save_path=save_path,
        load_path=Path(args.load_path) if args.load_path else None,
        resume=args.resume,
    )

    try:
        if load_path is None:
            model = DQN(
                args.policy,
                env,
                verbose=1,
                seed=args.seed,
                device=args.device,
            )
            reset_num_timesteps = True
            print("Starting a new DQN model.")
        else:
            model = DQN.load(load_path, env=env, device=args.device)
            if not args.no_save_replay_buffer:
                _load_replay_buffer_if_available(model, replay_buffer_path)
            reset_num_timesteps = args.reset_num_timesteps
            print(f"Loaded DQN model from {load_path}")

        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )
        model.save(save_path)
        if not args.no_save_replay_buffer:
            replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_replay_buffer(replay_buffer_path)
            print(f"Saved replay buffer to {replay_buffer_path}")
        print(f"Saved RL model to {save_path}")
    finally:
        env.close()


def _select_load_path(save_path: Path, load_path: Path | None, resume: bool) -> Path | None:
    if resume and save_path.exists():
        return save_path
    if load_path is not None:
        if not load_path.exists():
            raise FileNotFoundError(f"Requested load.path does not exist: {load_path}")
        return load_path
    if resume:
        print(f"load.resume is true, but {save_path} does not exist yet. Starting from scratch.")
    return None


def _load_replay_buffer_if_available(model: DQN, replay_buffer_path: Path) -> None:
    if replay_buffer_path.exists():
        model.load_replay_buffer(replay_buffer_path)
        print(f"Loaded replay buffer from {replay_buffer_path}")
    else:
        print(f"Replay buffer not found at {replay_buffer_path}; continuing without it.")


def _config_to_args(cfg) -> SimpleNamespace:
    return SimpleNamespace(
        cv_weights=cfg.cv.weights,
        cv_model_type=cfg.cv.model_type,
        cv_metadata=cfg.cv.metadata,
        save_path=cfg.output.save_path,
        load_path=cfg.load.path,
        resume=bool(cfg.load.resume),
        reset_num_timesteps=bool(cfg.load.reset_num_timesteps),
        timesteps=cfg.rl.timesteps,
        policy=cfg.rl.policy,
        seed=cfg.seed,
        device=cfg.device,
        obs_mode=cfg.env.obs_mode,
        visualize=bool(cfg.visualization.enabled),
        vis_dir=cfg.visualization.dir,
        vis_freq=cfg.visualization.freq,
        vis_max_steps=cfg.visualization.max_steps,
        vis_fps=cfg.visualization.fps,
        checkpoint_dir=cfg.checkpoint.dir,
        checkpoint_freq=cfg.checkpoint.freq,
        replay_buffer_path=cfg.replay_buffer.path,
        no_save_replay_buffer=not bool(cfg.replay_buffer.save),
    )


def _run_hydra_main() -> None:
    if hydra is None:
        raise SystemExit(
            "Hydra is required for config-driven RL training. "
            "Install dependencies with: pip install -r requirements.txt",
        )

    @hydra.main(version_base=None, config_path="configs/rl", config_name="train")
    def _main(cfg) -> None:
        run_training(_config_to_args(cfg))

    _main()


def _validate_args(args: SimpleNamespace) -> None:
    if args.cv_model_type not in CV_MODEL_TYPES:
        raise ValueError(f"cv.model_type must be one of {CV_MODEL_TYPES}, got {args.cv_model_type!r}.")
    if args.obs_mode not in ("hybrid", "cv-only"):
        raise ValueError("env.obs_mode must be either 'hybrid' or 'cv-only'.")
    if args.policy != "MlpPolicy":
        raise ValueError("Only rl.policy=MlpPolicy is supported by this training script.")
    if args.timesteps <= 0:
        raise ValueError("rl.timesteps must be positive.")
    if args.visualize and args.vis_freq <= 0:
        raise ValueError("visualization.freq must be positive.")
    if args.visualize and args.vis_max_steps < 0:
        raise ValueError("visualization.max_steps cannot be negative.")
    if args.visualize and args.vis_fps <= 0:
        raise ValueError("visualization.fps must be positive.")
    if args.checkpoint_freq < 0:
        raise ValueError("checkpoint.freq cannot be negative.")


def main() -> None:
    _run_hydra_main()


if __name__ == "__main__":
    main()
