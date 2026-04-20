"""
Последовательное обучение всех моделей ансамбля.
Запуск: python train_ensemble.py

Обучает модели для каждого цветового пространства из config.yaml:
    checkpoints/rgb/best_model.pth
    checkpoints/lab/best_model.pth
    checkpoints/hsv/best_model.pth
"""

import yaml
from train import train_one


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    colorspaces = cfg["ensemble"]["colorspaces"]
    results = {}

    print(f"Запускаем обучение ансамбля: {colorspaces}")
    print(f"Всего эпох на каждую модель: {cfg['train']['epochs']}")

    for colorspace in colorspaces:
        best_iou = train_one(cfg, colorspace)
        results[colorspace] = best_iou

    print("\n" + "="*55)
    print("  Итоги обучения ансамбля:")
    print("="*55)
    for cs, iou in results.items():
        print(f"  {cs.upper():<6}  IoU = {iou:.4f}")
    print("="*55)


if __name__ == "__main__":
    main()
