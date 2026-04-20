"""
Инференс с ансамблем всех обученных моделей + TTA.
Модели взвешиваются согласно config.yaml → ensemble → weights.

Запуск:
    python predict.py --img path/to/car.jpg

    # Если нужна только одна модель:
    python predict.py --img car.jpg --colorspace rgb
"""

import os
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as tfs_v2
from PIL import Image

from src.dataset import convert_colorspace, NORMALIZATIONS
from src.model   import UNetModel
from src.trainer import predict_with_tta


# ============================================================
#  Вспомогательные функции
# ============================================================

def get_val_transform(img_size: int, colorspace: str):
    norm = NORMALIZATIONS[colorspace]
    return tfs_v2.Compose([
        tfs_v2.Resize((img_size, img_size)),
        tfs_v2.ToImage(),
        tfs_v2.ToDtype(torch.float32, scale=True),
        tfs_v2.Normalize(mean=norm["mean"], std=norm["std"]),
    ])


def load_model(weights_path: str, device: torch.device) -> UNetModel:
    model = UNetModel(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval().to(device)
    return model


# ============================================================
#  Основная функция инференса
# ============================================================

def predict(img_path: str, cfg: dict, colorspace: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = cfg["data"]["img_size"]
    checkpoint_dir = cfg["paths"]["checkpoints"]

    # Определяем какие модели использовать
    if colorspace:
        # Одна конкретная модель
        colorspaces = [colorspace]
        weights_list = [1.0]
    else:
        # Полный ансамбль из конфига
        colorspaces  = cfg["ensemble"]["colorspaces"]
        weights_list = cfg["ensemble"]["weights"]

    img_rgb = Image.open(img_path).convert("RGB")
    predictions = {}

    # --- Предсказание каждой модели ---
    for cs, w in zip(colorspaces, weights_list):
        weights_path = os.path.join(checkpoint_dir, cs, "best_model.pth")
        if not os.path.exists(weights_path):
            print(f"[warn] веса не найдены: {weights_path}, пропускаем")
            continue

        print(f"Предсказание {cs.upper()} (вес={w})...")
        model = load_model(weights_path, device)
        tf    = get_val_transform(img_size, cs)

        img_converted = convert_colorspace(img_rgb, cs)
        tensor = tf(img_converted).unsqueeze(0)

        # predict_with_tta уже применяет sigmoid → [0, 1]
        pred = predict_with_tta(model, tensor, device)
        predictions[cs] = (pred, w)

        # Освобождаем память GPU
        del model
        torch.cuda.empty_cache()

    if not predictions:
        raise RuntimeError("Ни одна модель не загружена. Запусти train.py сначала.")

    # --- Взвешенный ансамбль ---
    total_weight = sum(w for _, w in predictions.values())
    ensemble = sum(pred * (w / total_weight) for pred, w in predictions.values())

    # --- Преобразование в numpy ---
    def to_mask(tensor):
        return np.clip(
            tensor.squeeze(0).permute(1, 2, 0).numpy() * 255, 0, 255
        ).astype("uint8")

    mask_ensemble = to_mask(ensemble)
    masks_individual = {cs: to_mask(pred) for cs, (pred, _) in predictions.items()}

    # --- Машина без фона ---
    img_display = np.array(img_rgb.resize((img_size, img_size)))
    car_only = img_display.copy()
    car_only[mask_ensemble.squeeze() < 127] = 0

    # --- Визуализация ---
    n_models = len(predictions)
    fig, axes = plt.subplots(2, n_models + 1, figsize=(5 * (n_models + 1), 10))

    # Верхний ряд: оригиналы в каждом цветовом пространстве
    for i, cs in enumerate(predictions.keys()):
        img_cs = convert_colorspace(img_rgb, cs)
        axes[0, i].imshow(img_cs.resize((img_size, img_size)))
        axes[0, i].set_title(f"Вход {cs.upper()}")
        axes[0, i].axis("off")

    axes[0, n_models].imshow(car_only)
    axes[0, n_models].set_title("Машина без фона")
    axes[0, n_models].axis("off")

    # Нижний ряд: маски каждой модели + итоговый ансамбль
    for i, (cs, mask) in enumerate(masks_individual.items()):
        w = predictions[cs][1]
        axes[1, i].imshow(mask.squeeze(), cmap="gray")
        axes[1, i].set_title(f"Маска {cs.upper()} (вес={w})")
        axes[1, i].axis("off")

    axes[1, n_models].imshow(mask_ensemble.squeeze(), cmap="gray")
    label = "Одна модель" if colorspace else f"Ансамбль ({'+'.join(predictions.keys())})"
    axes[1, n_models].set_title(label)
    axes[1, n_models].axis("off")

    plt.suptitle("Результаты сегментации", fontsize=14)
    plt.tight_layout()

    out_path = "result.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nСохранено: {out_path}")
    plt.show()


# ============================================================
#  Точка входа
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инференс ансамбля моделей сегментации")
    parser.add_argument("--img",        required=True, help="Путь к изображению")
    parser.add_argument("--colorspace", default=None,  choices=["rgb", "lab", "hsv"],
                        help="Если указан — использует только эту модель (default: весь ансамбль)")
    args = parser.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    predict(args.img, cfg, args.colorspace)
