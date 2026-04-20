"""
Веб-интерфейс для сегментации автомобилей.
Запуск: python app.py
Затем открыть: http://localhost:7860
"""

import os
import cv2
import numpy as np
import torch
import gradio as gr
import torchvision.transforms.v2 as tfs_v2
from PIL import Image

from src.model import UNetModel

# ============================================================
#  Настройки
# ============================================================

WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "checkpoints/rgb/best_model.pth")
IMG_SIZE     = 512
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLORSPACES = {
    "rgb":   {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "lab":   {"mean": [0.5, 0.5, 0.5],       "std": [0.5, 0.5, 0.5]},
    "hsv":   {"mean": [0.5, 0.5, 0.5],       "std": [0.5, 0.5, 0.5]},
    "ycrcb": {"mean": [0.5, 0.5, 0.5],       "std": [0.5, 0.5, 0.5]},
    "hls":   {"mean": [0.5, 0.5, 0.5],       "std": [0.5, 0.5, 0.5]},
}
CV2_CONV = {
    "rgb": None, "lab": cv2.COLOR_RGB2LAB,
    "hsv": cv2.COLOR_RGB2HSV, "ycrcb": cv2.COLOR_RGB2YCrCb,
    "hls": cv2.COLOR_RGB2HLS,
}

# ============================================================
#  Загрузка модели
# ============================================================

def load_model() -> UNetModel:
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Веса не найдены: {WEIGHTS_PATH}\n"
            "Положи best_model.pth в папку checkpoints/rgb/"
        )
    model = UNetModel(pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    return model.eval().to(DEVICE)


print(f"Загрузка модели, устройство: {DEVICE}")
model = load_model()
print("Модель готова!")

# ============================================================
#  Нормализация (работает с тензором напрямую)
# ============================================================

NORMALIZERS = {
    cs: tfs_v2.Normalize(mean=v["mean"], std=v["std"])
    for cs, v in COLORSPACES.items()
}


def img_to_tensor(img_rgb: Image.Image, cs: str) -> torch.Tensor:
    """
    PIL RGB → нормализованный тензор [1, 3, H, W].
    Конвертируем через numpy чтобы избежать проблем с PIL mode.
    """
    # Ресайз и конвертация в numpy uint8
    img_np = np.array(img_rgb.resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)

    if cs != "rgb":
        img_np = cv2.cvtColor(img_np, CV2_CONV[cs])

    # numpy → тензор float32 [0, 1]
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

    # Нормализация
    tensor = NORMALIZERS[cs](tensor)

    return tensor.unsqueeze(0)  # [1, 3, H, W]


# ============================================================
#  Инференс
# ============================================================

def predict_tta(tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        pred  = torch.sigmoid(model(tensor.to(DEVICE)))
        pf    = torch.sigmoid(model(torch.flip(tensor, dims=[3]).to(DEVICE)))
        pred  = (pred + torch.flip(pf, dims=[3])) / 2
    return pred.cpu()


def remove_noise(mask_np: np.ndarray, min_area_ratio: float) -> np.ndarray:
    if min_area_ratio <= 0:
        return mask_np
    total    = mask_np.shape[0] * mask_np.shape[1]
    min_area = int(total * min_area_ratio)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    clean = np.zeros_like(mask_np)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


def segment(
    input_image: Image.Image,
    threshold: float = 0.5,
    min_area: float = 0.01,
    use_tta: bool = True,
    colorspaces: list = None,
) -> tuple:
    if colorspaces is None:
        colorspaces = list(COLORSPACES.keys())

    img_rgb     = input_image.convert("RGB")
    img_display = np.array(img_rgb.resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)

    # Предсказания для каждого цветового пространства
    preds = []
    for cs in colorspaces:
        tensor = img_to_tensor(img_rgb, cs)
        if use_tta:
            pred = predict_tta(tensor)
        else:
            with torch.no_grad():
                pred = torch.sigmoid(model(tensor.to(DEVICE))).cpu()
        preds.append(pred)

    # Усреднение
    avg = sum(preds) / len(preds)

    # Бинаризация и очистка
    t = avg.squeeze()
    while t.dim() > 2:
        t = t.squeeze(0)
    mask_raw = (t.numpy() * 255).astype(np.uint8)
    _, binary = cv2.threshold(mask_raw, int(threshold * 255), 255, cv2.THRESH_BINARY)
    clean_mask = remove_noise(binary, min_area)

    # Машина на сером фоне
    gray_bg = np.full_like(img_display, 200, dtype=np.uint8)
    alpha   = clean_mask.astype(np.float32) / 255.0
    on_gray = (
        img_display * alpha[:, :, np.newaxis] +
        gray_bg * (1 - alpha[:, :, np.newaxis])
    ).astype(np.uint8)

    # PNG с прозрачным фоном (RGBA)
    rgba        = np.dstack([img_display, clean_mask])
    transparent = Image.fromarray(rgba.astype(np.uint8), mode="RGBA")

    return (
        Image.fromarray(clean_mask),
        Image.fromarray(on_gray),
        transparent,
    )


# ============================================================
#  Gradio интерфейс
# ============================================================

def run(image, threshold, min_area, use_tta, selected_cs):
    if image is None:
        return None, None, None
    cs_map = {
        "RGB": "rgb", "LAB": "lab", "HSV": "hsv",
        "YCrCb": "ycrcb", "HLS": "hls"
    }
    css = [cs_map[c] for c in selected_cs] if selected_cs else list(COLORSPACES.keys())
    return segment(image, threshold, min_area / 100, use_tta, css)


with gr.Blocks(title="Car Segmentation") as demo:
    gr.Markdown("""
    # 🚗 Car Segmentation
    Загрузи фото автомобиля — модель вернёт маску и изображение без фона.

    **Как работает:** одна модель (UNet + ResNet34) получает изображение
    в нескольких цветовых пространствах, предсказания усредняются.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            inp_image = gr.Image(type="pil", label="Входное изображение")

            gr.Markdown("### Параметры")
            threshold = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                label="Порог бинаризации",
                info="Выше = строже (меньше белых пикселей)"
            )
            min_area = gr.Slider(
                minimum=0, maximum=10, value=1, step=0.5,
                label="Минимальный размер объекта (%)",
                info="Убирает мелкий шум на фоне"
            )
            use_tta = gr.Checkbox(
                value=True, label="TTA (горизонтальный флип)",
                info="Немного медленнее, но точнее"
            )
            selected_cs = gr.CheckboxGroup(
                choices=["RGB", "LAB", "HSV", "YCrCb", "HLS"],
                value=["RGB", "LAB", "HSV", "YCrCb", "HLS"],
                label="Цветовые пространства для ансамбля",
            )
            btn = gr.Button("Сегментировать", variant="primary")

        with gr.Column(scale=2):
            out_mask        = gr.Image(label="Маска")
            out_gray        = gr.Image(label="Машина на сером фоне")
            out_transparent = gr.Image(label="Прозрачный фон (PNG)")

    btn.click(
        fn=run,
        inputs=[inp_image, threshold, min_area, use_tta, selected_cs],
        outputs=[out_mask, out_gray, out_transparent],
    )

    gr.Markdown("""
    ---
    **Модель:** UNet + ResNet34 (предобучен на ImageNet) · **Val IoU: 0.89**  
    **Датасет:** [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge)
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
