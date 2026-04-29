# RL-checkpoint'ы

Здесь хранятся checkpoint'ы обученных reinforcement-learning агентов.

Рекомендуемая структура:

```text
checkpoints/rl/<agent_version>/
```

Текущие артефакты моделей Stable-Baselines3 DQN находятся здесь:

```text
checkpoints/rl/sb3_dqn/models/
```

Периодические checkpoint'ы обучения находятся здесь:

```text
checkpoints/rl/sb3_dqn/periodic/
```

Директория `runs/` оставлена для визуализаций и других логов запусков.

SB3-архивы, replay buffer'ы и run-логи являются локальными артефактами и не
коммитятся в git.
