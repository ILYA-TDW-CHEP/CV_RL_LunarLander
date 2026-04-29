# CV-RL LunarLander

## Git-политика

В git должны попадать исходный код, notebooks, README, JSON metadata и конфиги
проекта. Локальные данные и артефакты обучения не коммитятся:

- `data/images/` и `data/labels.csv` генерируются через
  `notebooks/00_data/01_generate_images.ipynb`;
- checkpoint'ы моделей (`*.pth`, `*.pt`, `*.onnx`), SB3-архивы (`*.zip`) и
  replay buffer'ы (`*.pkl`) остаются локально;
- `runs/`, GIF/PNG/CSV визуализаций, Python/Jupyter-кэши и виртуальные
  окружения исключены через `.gitignore`.

После клонирования проекта сначала установите зависимости, затем пересоздайте
датасет через notebook генерации, если он нужен для обучения CV.

Этот проект объединяет компьютерное зрение и reinforcement learning для
`LunarLander-v3`. Ноутбуки генерируют отрендеренные кадры LunarLander и обучают
ResNet18-регрессор, который предсказывает позу аппарата по изображениям.

Текущий checkpoint
`checkpoints/cv/resnet18_pose/state_regressor_resnet18.pth` предсказывает:

```text
[x, y, sin(theta), cos(theta)]
```

RL-wrapper рендерит каждый кадр симулятора, пропускает его через CV-модель и
использует предсказанное состояние как наблюдение для RL-агента.

## Установка

```bash
pip install -r requirements.txt
pip install -e .
```

Для LunarLander в Gymnasium нужен extra-пакет Box2D. На некоторых системах
могут также понадобиться системные инструменты сборки.

## CV-модель

В репозитории уже есть
`checkpoints/cv/resnet18_pose/state_regressor_resnet18.pth`. Этот checkpoint
соответствует архитектуре `StateRegressorResNet18` из
`src/lunar_lander_cvrl/vision.py`.

Пример базового инференса:

```bash
python -c "import numpy as np; from lunar_lander_cvrl import StatePredictor; p=StatePredictor('checkpoints/cv/resnet18_pose/state_regressor_resnet18.pth', device='cpu'); print(p.predict_pose(np.load('data/images/frame_0000000.npy')))"
```

### Обучение CV-регрессора

`train_cv.py` обучает CV-регрессор по выбранной интеграции и типу модели.
Интеграция задаётся через `data/cv_integrations/<name>/metadata.json`: там
указаны общие `images_dir`, `labels_file` и нужные `target_columns`.

```bash
python train_cv.py \
  --integration x_y_theta \
  --model-type resnet18 \
  --version resnet18_pose_v2 \
  --epochs 20 \
  --batch-size 32 \
  --device cpu
```

Доступные типы моделей:

- `resnet18`: ResNet18-регрессор, совместимый с текущим baseline.
- `simple-cnn`: компактная CNN для быстрых экспериментов.

Результаты сохраняются в `checkpoints/cv/<version>/`: веса `model.pth`, история
обучения `history.json` и конфиг запуска `training_config.json`.

## Обучение RL-агента

Режим по умолчанию — `hybrid`: CV предсказывает `x`, `y` и `theta`, а
Gymnasium отдаёт настоящие скорости и контакты ног.

```bash
python train_rl.py \
  --cv-weights checkpoints/cv/resnet18_pose/state_regressor_resnet18.pth \
  --save-path checkpoints/rl/sb3_dqn/models/dqn_vision_lander.zip \
  --timesteps 100000 \
  --seed 42 \
  --device cpu \
  --obs-mode hybrid
```

Для более строгого CV-only эксперимента:

```bash
python train_rl.py \
  --cv-weights checkpoints/cv/resnet18_pose/state_regressor_resnet18.pth \
  --save-path checkpoints/rl/sb3_dqn/models/dqn_vision_lander_cv_only.zip \
  --timesteps 100000 \
  --seed 42 \
  --device cpu \
  --obs-mode cv-only
```

В режиме `cv-only` скорости оцениваются конечными разностями, а контакты ног
задаются как `0.0`.

### Обучение в несколько сессий

Используйте `--resume`, чтобы продолжить обучение модели из `--save-path`.
Каждый запуск добавляет ещё `--timesteps` шагов обучения, а затем перезаписывает
`--save-path` обновлённой моделью.

```bash
python train_rl.py \
  --cv-weights checkpoints/cv/resnet18_pose/state_regressor_resnet18.pth \
  --save-path checkpoints/rl/sb3_dqn/models/dqn_vision_lander.zip \
  --timesteps 200000 \
  --seed 42 \
  --device cpu \
  --obs-mode hybrid \
  --resume \
  --checkpoint-freq 50000
```

Для следующей сессии запустите ту же команду ещё раз. Периодические checkpoint'ы
сохраняются в `checkpoints/rl/sb3_dqn/periodic/`; используйте
`--checkpoint-freq 0`, чтобы отключить их. Для DQN скрипт также сохраняет replay
buffer рядом с моделью, например
`checkpoints/rl/sb3_dqn/models/dqn_vision_lander.replay_buffer.pkl`, и
загружает его при следующем запуске с `--resume`.

### Визуализация обучения

Добавьте `--visualize`, чтобы периодически рендерить текущую политику во время
обучения. Скрипт сохраняет GIF-rollout'ы в `runs/visualizations/episodes/`,
записывает `runs/visualizations/training_visualization.csv` и обновляет
`runs/visualizations/training_rewards.png`.
По умолчанию каждый GIF записывает полный эпизод, пока Gymnasium не вернёт
`terminated` или `truncated`; используйте `--vis-max-steps N` только если явно
нужен более короткий preview. Финальный GIF всегда сохраняется в конце обучения.

```bash
python train_rl.py \
  --cv-weights checkpoints/cv/resnet18_pose/state_regressor_resnet18.pth \
  --save-path checkpoints/rl/sb3_dqn/models/dqn_vision_lander.zip \
  --timesteps 100000 \
  --seed 42 \
  --device cpu \
  --obs-mode hybrid \
  --visualize \
  --vis-freq 10000
```

## Оценка

```bash
python evaluate_rl.py \
  --cv-weights checkpoints/cv/resnet18_pose/state_regressor_resnet18.pth \
  --model-path checkpoints/rl/sb3_dqn/models/dqn_vision_lander.zip \
  --episodes 10 \
  --seed 100 \
  --device cpu \
  --obs-mode hybrid
```

Оценка выводит среднюю награду, стандартное отклонение награды и количество
успешных посадок. По умолчанию эпизод с reward `>= 200` считается успешным.

## Заметки

- `notebooks/00_data/01_generate_images.ipynb` генерирует датасет отрендеренных
  кадров и метки.
- `notebooks/01_cv/01_train_resnet_pose.ipynb` обучает и оценивает CV-checkpoint
  ResNet18.
- `data/images/` и `data/labels.csv` являются общим источником данных; варианты
  CV-интеграции задают только `metadata.json` с нужными `target_columns`.
- Доступные supervised-метки содержат только позу, без скоростей, контактов,
  действий, наград и done-флагов.
- Качество RL зависит от точности и обобщающей способности CV-модели.
- Для полезной политики может потребоваться много шагов обучения.
