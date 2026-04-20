"""
Обучение одной модели для заданного цветового пространства.

Запуск:
    python train.py                     # обучает RGB (по умолчанию)
    python train.py --colorspace lab    # обучает LAB
    python train.py --colorspace hsv    # обучает HSV

Для обучения всех трёх сразу используй train_ensemble.py
"""

import argparse
import yaml
import torch
import torch.optim as optim
import torch.utils.data as data

from src.dataset import CarvanaDataset, get_transforms, COLORSPACES
from src.model   import UNetModel
from src.loss    import CombinedLoss
from src.trainer import Trainer


def train_one(cfg: dict, colorspace: str):
    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"\n{'='*55}")
    print(f"  Цветовое пространство : {colorspace.upper()}")
    print(f"  Устройство            : {device}")
    if device.type == "cuda":
        print(f"  GPU                   : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM                  : {vram:.1f} GB")
    print(f"{'='*55}\n")

    # --- Трансформы с нормализацией под colorspace ---
    joint_tf, img_tf, mask_tf, val_img_tf, val_mask_tf = get_transforms(
        cfg["data"]["img_size"], colorspace
    )

    # --- Датасеты ---
    d_train = CarvanaDataset(
        root=cfg["data"]["root"],
        split="train",
        val_ratio=cfg["data"]["val_ratio"],
        seed=cfg["data"]["seed"],
        colorspace=colorspace,
        joint_transform=joint_tf,
        img_transform=img_tf,
        mask_transform=mask_tf,
    )
    d_val = CarvanaDataset(
        root=cfg["data"]["root"],
        split="val",
        val_ratio=cfg["data"]["val_ratio"],
        seed=cfg["data"]["seed"],
        colorspace=colorspace,
        joint_transform=None,
        img_transform=val_img_tf,
        mask_transform=val_mask_tf,
    )

    train_loader = data.DataLoader(
        d_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = data.DataLoader(
        d_val,
        batch_size=cfg["train"]["val_batch_size"],
        shuffle=False,
        num_workers=max(cfg["train"]["num_workers"] // 2, 1),
        pin_memory=True,
        persistent_workers=True,
    )

    # --- Модель ---
    # RGB: используем предобученные веса ImageNet
    # LAB/HSV: обучаем энкодер с нуля (pretrained=False)
    pretrained = (colorspace == "rgb")
    model = UNetModel(pretrained=pretrained)

    # Для LAB/HSV — энкодер тоже учится активно (lr выше)
    lr_enc = cfg["train"]["lr_encoder"] if pretrained else cfg["train"]["lr_decoder"]

    optimizer = optim.AdamW(
        model.get_param_groups(
            lr_encoder=lr_enc,
            lr_decoder=cfg["train"]["lr_decoder"],
            weight_decay=cfg["train"]["weight_decay"],
        ),
        weight_decay=cfg["train"]["weight_decay"],
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg["scheduler"]["mode"],
        factor=cfg["scheduler"]["factor"],
        patience=cfg["scheduler"]["patience"],
    )

    loss_fn = CombinedLoss(boundary_weight=cfg["loss"]["boundary_weight"])

    # Веса каждой модели в отдельной папке
    checkpoint_dir = f"{cfg['paths']['checkpoints']}/{colorspace}"

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=checkpoint_dir,
        consistency_weight=cfg["consistency"]["weight"],
        view_weights=tuple(cfg["consistency"]["view_weights"]),
    )

    best_iou = trainer.fit(train_loader, val_loader, cfg["train"]["epochs"])
    return best_iou


def main():
    parser = argparse.ArgumentParser(description="Обучение модели сегментации")
    parser.add_argument(
        "--colorspace",
        default="rgb",
        choices=COLORSPACES,
        help="Цветовое пространство для обучения (default: rgb)",
    )
    args = parser.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    train_one(cfg, args.colorspace)


if __name__ == "__main__":
    main()
