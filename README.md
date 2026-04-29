# CV-RL LunarLander

Проект для экспериментов на стыке computer vision и reinforcement learning в
среде `LunarLander-v3`. Идея пайплайна: CV-модель получает RGB-кадр симулятора,
предсказывает состояние аппарата, а RL-агент использует это состояние как
наблюдение для управления посадкой.

Проект сделан как пет-проект с возможностью менять обе части:

- CV-регрессор: можно выбирать архитектуру и набор target-полей через
  `train_cv.py` и `data/cv_integrations/*/metadata.json`.
- RL-агент: текущий CLI обучает Stable-Baselines3 DQN, а структура
  `src/lunar_lander_cvrl/models/rl/` и `checkpoints/rl/` подготовлена под
  будущие версии агентов.

## Структура

```text
data/
  images/                  # локально генерируемые кадры, не коммитятся
  labels.csv               # локально генерируемые метки, не коммитятся
  cv_integrations/         # JSON-конфиги target-полей для CV

notebooks/
  00_data/                 # генерация данных
  01_cv/                   # CV-эксперименты
  02_rl/                   # RL-эксперименты
  03_experiments/          # смешанные эксперименты и отчёты

src/lunar_lander_cvrl/
  envs.py                  # Gymnasium wrapper для CV-derived observations
  vision.py                # инференс CV-модели
  models/cv/               # версии CV-моделей
  models/rl/               # версии RL-моделей

checkpoints/
  cv/                      # локальные CV checkpoint'ы
  rl/                      # локальные RL checkpoint'ы
```

## Установка

```bash
pip install -r requirements.txt
pip install -e .
```

Для `LunarLander-v3` в Gymnasium используется Box2D. На некоторых системах для
него могут понадобиться дополнительные системные инструменты сборки.

## Данные

Кадры и метки генерируются локально и не хранятся в git:

- `data/images/`
- `data/labels.csv`

Чтобы создать датасет, запустите notebook:

```text
notebooks/00_data/01_generate_images.ipynb
```

Варианты CV-интеграции задаются через JSON-конфиги в
`data/cv_integrations/`. Например, `x_y` берёт только координаты, а
`x_y_theta` берёт координаты и угол. Dataloader читает общий `labels.csv` и
использует только колонки из `target_columns`.

## Обучение CV

`train_cv.py` обучает CV-регрессор по выбранной интеграции и типу модели.

Пример:

```bash
python train_cv.py \
  --integration x_y_theta \
  --model-type resnet18 \
  --version cv_pose_v1 \
  --epochs 20 \
  --batch-size 32 \
  --device cpu
```

Доступные типы CV-моделей:

- `resnet18`
- `simple-cnn`

Результаты сохраняются в:

```text
checkpoints/cv/<version>/
```

В папке версии сохраняются веса `model.pth`, история обучения `history.json` и
конфиг запуска `training_config.json`.

## CV-инференс

Пример инференса с обученным checkpoint'ом:

```bash
python -c "import numpy as np; from lunar_lander_cvrl import StatePredictor; p=StatePredictor('checkpoints/cv/resnet18_pose/state_regressor_resnet18.pth', device='cpu'); print(p.predict_pose(np.load('data/images/frame_0000000.npy')))"
```

Для новых CV-версий можно передавать другой путь к весам. Если архитектура
отличается от стандартной, используйте соответствующую модель при создании
`StatePredictor` или добавьте отдельный predictor под новую версию.

## Обучение RL

`train_rl.py` обучает RL-агента на наблюдениях, полученных через CV-wrapper.
Текущая реализация использует Stable-Baselines3 DQN.

```bash
python train_rl.py \
  --cv-weights checkpoints/cv/resnet18_pose/state_regressor_resnet18.pth \
  --cv-model-type resnet18 \
  --cv-metadata data/cv_integrations/x_y_theta/metadata.json \
  --save-path checkpoints/rl/sb3_dqn/models/dqn_vision_lander.zip \
  --timesteps 100000 \
  --seed 42 \
  --device cpu \
  --obs-mode hybrid
```

Режимы наблюдений:

- `hybrid`: CV предсказывает `x`, `y`, `theta`, а Gymnasium отдаёт скорости и
  контакты ног. Если CV-модель обучена только на `x_y`, то `hybrid` берёт
  `x`, `y` из CV, а `theta` и остальные компоненты из Gymnasium.
- `cv-only`: скорости оцениваются конечными разностями, контакты ног задаются
  как `0.0`. Этот режим требует, чтобы CV-модель предсказывала `theta` или
  `sin_theta/cos_theta`.

Пример RL с CV-моделью без `theta`:

```bash
python train_rl.py \
  --cv-weights checkpoints/cv/x_y_model/model.pth \
  --cv-model-type simple-cnn \
  --cv-metadata data/cv_integrations/x_y/metadata.json \
  --save-path checkpoints/rl/sb3_dqn/models/dqn_xy_hybrid.zip \
  --timesteps 100000 \
  --seed 42 \
  --device cpu \
  --obs-mode hybrid
```

Для продолжения обучения используйте `--resume`:

```bash
python train_rl.py \
  --cv-weights checkpoints/cv/resnet18_pose/state_regressor_resnet18.pth \
  --cv-model-type resnet18 \
  --cv-metadata data/cv_integrations/x_y_theta/metadata.json \
  --save-path checkpoints/rl/sb3_dqn/models/dqn_vision_lander.zip \
  --timesteps 200000 \
  --seed 42 \
  --device cpu \
  --obs-mode hybrid \
  --resume
```

Новые версии RL-агентов можно добавлять в `src/lunar_lander_cvrl/models/rl/` и
сохранять их артефакты в `checkpoints/rl/<agent_version>/`.

## Визуализация RL

Добавьте `--visualize`, чтобы периодически сохранять rollout'ы текущей политики:

```bash
python train_rl.py \
  --cv-weights checkpoints/cv/resnet18_pose/state_regressor_resnet18.pth \
  --cv-model-type resnet18 \
  --cv-metadata data/cv_integrations/x_y_theta/metadata.json \
  --save-path checkpoints/rl/sb3_dqn/models/dqn_vision_lander.zip \
  --timesteps 100000 \
  --seed 42 \
  --device cpu \
  --obs-mode hybrid \
  --visualize \
  --vis-freq 10000
```

Визуализации и run-логи сохраняются в `runs/` и не коммитятся в git.

## Оценка RL

```bash
python evaluate_rl.py \
  --cv-weights checkpoints/cv/resnet18_pose/state_regressor_resnet18.pth \
  --cv-model-type resnet18 \
  --cv-metadata data/cv_integrations/x_y_theta/metadata.json \
  --model-path checkpoints/rl/sb3_dqn/models/dqn_vision_lander.zip \
  --episodes 10 \
  --seed 100 \
  --device cpu \
  --obs-mode hybrid
```

Скрипт выводит награду по эпизодам, среднюю награду, стандартное отклонение и
число успешных посадок.
