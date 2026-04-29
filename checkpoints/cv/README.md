# CV-checkpoint'ы

Здесь хранятся checkpoint'ы обученных computer-vision моделей.

Рекомендуемая структура:

```text
checkpoints/cv/<model_version>/
```

Текущий checkpoint ResNet18 для позы находится здесь:

```text
checkpoints/cv/resnet18_pose/state_regressor_resnet18.pth
```

Файлы весов (`*.pth`, `*.pt`, `*.onnx`) являются локальными артефактами и не
коммитятся в git.
