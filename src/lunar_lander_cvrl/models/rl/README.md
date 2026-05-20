# RL models

The main training entrypoint is `train_rl.py`. It supports Stable-Baselines3
`DQN` and `PPO` through the Hydra field `rl.algorithm`.

Useful configs:

- `configs/rl/train.yaml` - DQN defaults.
- `configs/rl/train_ppo.yaml` - PPO defaults.

Shared SB3 helpers live in `utils.py`: argument validation, algorithm lookup,
model construction, checkpoint loading and replay-buffer persistence.
