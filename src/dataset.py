import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs_v2


# ============================================================
#  Цветовые пространства
# ============================================================

COLORSPACES = ["rgb", "lab", "hsv"]

# Нормализация под каждое пространство:
# RGB — ImageNet статистики (энкодер предобучен на ImageNet)
# LAB/HSV — центрирование в [-1, 1] т.к. предобученных весов нет
NORMALIZATIONS = {
    "rgb": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "lab": {"mean": [0.5, 0.5, 0.5],       "std": [0.5, 0.5, 0.5]},
    "hsv": {"mean": [0.5, 0.5, 0.5],       "std": [0.5, 0.5, 0.5]},
}


def convert_colorspace(img: Image.Image, mode: str) -> Image.Image:
    """Конвертирует PIL RGB изображение в нужное цветовое пространство"""
    if mode == "rgb":
        return img
    img_np = np.array(img)
    if mode == "lab":
        converted = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    elif mode == "hsv":
        converted = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    else:
        raise ValueError(f"Неизвестное цветовое пространство: {mode}. Доступны: {COLORSPACES}")
    return Image.fromarray(converted)


# ============================================================
#  Датасет
# ============================================================

class CarvanaDataset(data.Dataset):
    """
    root/
      train/         <- abc123_01.jpg
      train_masks/   <- abc123_01_mask.gif

    colorspace: "rgb" | "lab" | "hsv"
        Изображение конвертируется ДО применения трансформов.
        Маска всегда остаётся в grayscale — цветовое пространство на неё не влияет.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        val_ratio: float = 0.1,
        colorspace: str = "rgb",
        joint_transform=None,
        img_transform=None,
        mask_transform=None,
        seed: int = 42,
    ):
        if colorspace not in COLORSPACES:
            raise ValueError(f"colorspace должен быть одним из {COLORSPACES}")

        self.root = Path(root)
        self.colorspace = colorspace
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        img_dir = self.root / "train"
        mask_dir = self.root / "train_masks"

        pairs = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            mask_path = mask_dir / (img_path.stem + "_mask.gif")
            if mask_path.exists():
                pairs.append((img_path, mask_path))
            else:
                print(f"[warn] маска не найдена: {mask_path}")

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(pairs))
        split_at = int(len(pairs) * (1 - val_ratio))
        selected = indices[:split_at] if split == "train" else indices[split_at:]
        self.pairs = [pairs[i] for i in selected]
        print(f"[{colorspace.upper()}][{split}] пар: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Конвертируем цветовое пространство ДО трансформов
        img = convert_colorspace(img, self.colorspace)

        if self.joint_transform:
            img, mask = self.joint_transform(img, mask)
        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = (mask >= 127.0 / 255.0).float()
        return img, mask


# ============================================================
#  Трансформации
# ============================================================

def get_transforms(img_size: int, colorspace: str):
    """
    Возвращает (joint, img, mask, val_img, val_mask) трансформы
    с нормализацией под конкретное цветовое пространство.
    """
    size = (img_size, img_size)
    norm = NORMALIZATIONS[colorspace]

    joint = tfs_v2.Compose([
        tfs_v2.Resize(size),
        tfs_v2.RandomHorizontalFlip(p=0.5),
        tfs_v2.RandomRotation(degrees=10),
        tfs_v2.RandomResizedCrop(size=size, scale=(0.8, 1.0), antialias=True),
    ])

    img = tfs_v2.Compose([
        tfs_v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        tfs_v2.RandomGrayscale(p=0.1),
        tfs_v2.ToImage(),
        tfs_v2.ToDtype(torch.float32, scale=True),
        tfs_v2.Normalize(mean=norm["mean"], std=norm["std"]),
    ])

    mask = tfs_v2.Compose([
        tfs_v2.ToImage(),
        tfs_v2.ToDtype(torch.float32),
    ])

    val_img = tfs_v2.Compose([
        tfs_v2.Resize(size),
        tfs_v2.ToImage(),
        tfs_v2.ToDtype(torch.float32, scale=True),
        tfs_v2.Normalize(mean=norm["mean"], std=norm["std"]),
    ])

    val_mask = tfs_v2.Compose([
        tfs_v2.Resize(size),
        tfs_v2.ToImage(),
        tfs_v2.ToDtype(torch.float32),
    ])

    return joint, img, mask, val_img, val_mask
