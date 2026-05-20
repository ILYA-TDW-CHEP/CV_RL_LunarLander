"""Train an SB3 agent on CV-derived LunarLander observations."""

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
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:  # pragma: no cover - depends on optional package.
    raise SystemExit(
        "stable-baselines3 is required for RL training. "
        "Install dependencies with: pip install -r requirements.txt",
    ) from exc

from lunar_lander_cvrl.envs import make_vision_lander_env
from lunar_lander_cvrl.models.rl.utils import (
    create_sb3_model,
    load_replay_buffer_if_available,
    load_sb3_model,
    resolve_replay_buffer_path,
    save_model_artifacts,
    select_load_path,
    validate_training_args,
)
from lunar_lander_cvrl.visualization import TrainingVisualizationCallback


def run_training(args: SimpleNamespace) -> None:
    validate_training_args(args)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    replay_buffer_path = resolve_replay_buffer_path(save_path, args.replay_buffer_path)

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
                overlay_cv_pose=args.overlay_cv_pose,
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
                save_replay_buffer=args.algorithm == "dqn" and not args.no_save_replay_buffer,
                save_vecnormalize=True,
            ),
        )

    callback = CallbackList(callbacks) if callbacks else None

    load_path = select_load_path(
        save_path=save_path,
        load_path=Path(args.load_path) if args.load_path else None,
        resume=args.resume,
    )

    try:
        if load_path is None:
            model = create_sb3_model(args, env)
            reset_num_timesteps = True
            print(f"Starting a new {args.algorithm.upper()} model.")
        else:
            model = load_sb3_model(args, env, load_path)
            if args.algorithm == "dqn" and not args.no_save_replay_buffer:
                load_replay_buffer_if_available(model, replay_buffer_path)
            reset_num_timesteps = args.reset_num_timesteps
            print(f"Loaded {args.algorithm.upper()} model from {load_path}")

        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )
        save_model_artifacts(
            model=model,
            algorithm=args.algorithm,
            save_path=save_path,
            replay_buffer_path=replay_buffer_path,
            save_replay_buffer=not args.no_save_replay_buffer,
        )
    finally:
        env.close()


def _config_to_args(cfg) -> SimpleNamespace:
    return SimpleNamespace(
        algorithm=cfg.rl.algorithm,
        cv_weights=cfg.cv.weights,
        cv_model_type=cfg.cv.model_type,
        cv_metadata=cfg.cv.metadata,
        save_path=cfg.output.save_path,
        load_path=cfg.load.path,
        resume=bool(cfg.load.resume),
        reset_num_timesteps=bool(cfg.load.reset_num_timesteps),
        timesteps=cfg.rl.timesteps,
        policy=cfg.rl.policy,
        gamma=cfg.rl.gamma,
        dqn_learning_rate=cfg.rl.dqn.learning_rate,
        dqn_buffer_size=cfg.rl.dqn.buffer_size,
        dqn_learning_starts=cfg.rl.dqn.learning_starts,
        dqn_batch_size=cfg.rl.dqn.batch_size,
        dqn_train_freq=cfg.rl.dqn.train_freq,
        dqn_gradient_steps=cfg.rl.dqn.gradient_steps,
        dqn_target_update_interval=cfg.rl.dqn.target_update_interval,
        dqn_exploration_fraction=cfg.rl.dqn.exploration_fraction,
        dqn_exploration_final_eps=cfg.rl.dqn.exploration_final_eps,
        ppo_learning_rate=cfg.rl.ppo.learning_rate,
        ppo_n_steps=cfg.rl.ppo.n_steps,
        ppo_batch_size=cfg.rl.ppo.batch_size,
        ppo_n_epochs=cfg.rl.ppo.n_epochs,
        ppo_gae_lambda=cfg.rl.ppo.gae_lambda,
        ppo_clip_range=cfg.rl.ppo.clip_range,
        ppo_ent_coef=cfg.rl.ppo.ent_coef,
        ppo_vf_coef=cfg.rl.ppo.vf_coef,
        seed=cfg.seed,
        device=cfg.device,
        obs_mode=cfg.env.obs_mode,
        visualize=bool(cfg.visualization.enabled),
        vis_dir=cfg.visualization.dir,
        vis_freq=cfg.visualization.freq,
        vis_max_steps=cfg.visualization.max_steps,
        vis_fps=cfg.visualization.fps,
        overlay_cv_pose=bool(cfg.visualization.overlay_cv_pose),
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


def main() -> None:
    _run_hydra_main()


if __name__ == "__main__":
    main()
