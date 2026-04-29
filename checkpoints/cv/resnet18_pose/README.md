# Checkpoint ResNet18 для позы

В этой папке хранится текущий checkpoint ResNet18-регрессора позы.

Текущий артефакт:

- `state_regressor_resnet18.pth`

Модель предсказывает `[x, y, sin(theta), cos(theta)]` по отрендеренным кадрам
LunarLander.
