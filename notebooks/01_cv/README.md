# CV-ноутбуки

Используйте эту папку для computer-vision экспериментов, которые предсказывают
состояние LunarLander по отрендеренным кадрам.

`01_train_resnet_pose.ipynb` обучает и оценивает текущий ResNet18-регрессор
позы. Он читает конфиг интеграции
`../../data/cv_integrations/x_y_theta/metadata.json`, берёт нужные
`target_columns` из общего `../../data/labels.csv` и сохраняет веса в
`../../checkpoints/cv/resnet18_pose/state_regressor_resnet18.pth`.

Для воспроизводимого CLI-обучения используйте `train_cv.py` из корня проекта.
Через флаги можно выбрать интеграцию, версию checkpoint'а и тип регрессора.

Идеи для будущих ноутбуков:

- baseline-регрессия позы
- собственные CNN-архитектуры
- варианты target'ов из `data/cv_integrations`
- оценка CV-checkpoint'ов
