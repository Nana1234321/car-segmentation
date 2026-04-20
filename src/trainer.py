import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as tfs_v2
from tqdm import tqdm


# ============================================================
#  Метрики
# ============================================================

def iou_score(predict, target, threshold=0.5):
    """Intersection over Union — основная метрика сегментации"""
    predict = (torch.sigmoid(predict) > threshold).float()
    intersection = (predict * target).sum(dim=(1, 2, 3))
    union = (predict + target - predict * target).sum(dim=(1, 2, 3))
    return ((intersection + 1) / (union + 1)).mean().item()


# ============================================================
#  TTA
# ============================================================

def predict_with_tta(model, img_tensor, device):
    """Усредняет предсказания оригинала и горизонтального флипа"""
    model.eval()
    with torch.no_grad():
        pred_orig    = torch.sigmoid(model(img_tensor.to(device)))
        img_flipped  = torch.flip(img_tensor, dims=[3])
        pred_flipped = torch.sigmoid(model(img_flipped.to(device)))
        pred_flipped = torch.flip(pred_flipped, dims=[3])
    return ((pred_orig + pred_flipped) / 2).cpu()


# ============================================================
#  Конвертация батча RGB → LAB / HSV
# ============================================================

_LAB_NORM = tfs_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
_HSV_NORM = tfs_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
_RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_RGB_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def batch_to_colorspace(x_rgb: torch.Tensor, colorspace: str) -> torch.Tensor:
    """
    Конвертирует нормализованный RGB батч в LAB или HSV.
    x_rgb: [B, 3, H, W] с ImageNet нормализацией
    Возвращает: [B, 3, H, W] нормализованный в [-1, 1]
    """
    device = x_rgb.device
    x_denorm = (x_rgb.cpu() * _RGB_STD + _RGB_MEAN).clamp(0, 1)

    result = []
    for i in range(x_denorm.shape[0]):
        img_np = (x_denorm[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if colorspace == "lab":
            converted = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            norm_fn   = _LAB_NORM
        elif colorspace == "hsv":
            converted = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            norm_fn   = _HSV_NORM
        else:
            raise ValueError(f"Неизвестное пространство: {colorspace}")
        result.append(torch.from_numpy(converted).permute(2, 0, 1).float() / 255.0)

    return norm_fn(torch.stack(result)).to(device)


# ============================================================
#  Trainer
# ============================================================

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        checkpoint_dir,
        consistency_weight: float = 0.1,
        view_weights: tuple = (0.5, 0.3, 0.2),
    ):
        """
        consistency_weight:
            0.0 — обычное обучение без consistency
            0.1 — лёгкая регуляризация (рекомендуется для старта)
            0.3 — сильная регуляризация

        view_weights:
            (w_rgb, w_lab, w_hsv) — веса при усреднении предсказаний.
            RGB получает больший вес т.к. модель обучена на ImageNet (RGB).
        """
        self.model              = model.to(device)
        self.optimizer          = optimizer
        self.scheduler          = scheduler
        self.loss_fn            = loss_fn
        self.device             = device
        self.checkpoint_dir     = checkpoint_dir
        self.consistency_weight = consistency_weight
        self.view_weights       = view_weights
        self.best_iou           = 0.0
        self.scaler             = torch.amp.GradScaler("cuda")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"Trainer инициализирован:")
        print(f"  consistency_weight = {consistency_weight}")
        print(f"  view_weights (rgb/lab/hsv) = {view_weights}")

    # ----------------------------------------------------------
    #  Multi-view: три прогона одного батча
    # ----------------------------------------------------------

    def _multi_view_predict(self, x_rgb: torch.Tensor):
        """
        Прогоняет батч через модель в RGB, LAB и HSV.
        Возвращает логиты и взвешенное среднее вероятностей.
        """
        w_rgb, w_lab, w_hsv = self.view_weights

        x_lab = batch_to_colorspace(x_rgb, "lab")
        x_hsv = batch_to_colorspace(x_rgb, "hsv")

        with torch.amp.autocast("cuda"):
            pred_rgb = self.model(x_rgb)
            pred_lab = self.model(x_lab)
            pred_hsv = self.model(x_hsv)

        p_rgb = torch.sigmoid(pred_rgb)
        p_lab = torch.sigmoid(pred_lab)
        p_hsv = torch.sigmoid(pred_hsv)
        pred_avg = p_rgb * w_rgb + p_lab * w_lab + p_hsv * w_hsv

        return pred_rgb, pred_lab, pred_hsv, pred_avg

    def _consistency_loss(self, p_rgb, p_lab, p_hsv):
        """
        MSE каждого предсказания относительно взвешенного среднего.
        avg.detach() — не обновляем среднее как цель, только как сигнал.
        """
        w_rgb, w_lab, w_hsv = self.view_weights
        avg = (p_rgb * w_rgb + p_lab * w_lab + p_hsv * w_hsv).detach()
        return (
            nn.functional.mse_loss(p_rgb, avg) +
            nn.functional.mse_loss(p_lab, avg) +
            nn.functional.mse_loss(p_hsv, avg)
        )

    # ----------------------------------------------------------
    #  Train epoch
    # ----------------------------------------------------------

    def _train_epoch(self, loader, epoch, epochs):
        self.model.train()
        loss_mean    = 0
        main_mean    = 0
        consist_mean = 0
        lm_count     = 0

        pbar = tqdm(loader, leave=True, desc=f"Epoch [{epoch}/{epochs}] train")
        for x_rgb, y in pbar:
            x_rgb = x_rgb.to(self.device)
            y     = y.to(self.device)

            self.optimizer.zero_grad()

            if self.consistency_weight > 0:
                # Три прогона: RGB + LAB + HSV
                pred_rgb, pred_lab, pred_hsv, _ = self._multi_view_predict(x_rgb)

                with torch.amp.autocast("cuda"):
                    main_loss = self.loss_fn(pred_rgb, y)
                    p_rgb = torch.sigmoid(pred_rgb)
                    p_lab = torch.sigmoid(pred_lab)
                    p_hsv = torch.sigmoid(pred_hsv)
                    c_loss = self._consistency_loss(p_rgb, p_lab, p_hsv)
                    loss   = main_loss + self.consistency_weight * c_loss
            else:
                # Обычное обучение — один прогон
                with torch.amp.autocast("cuda"):
                    pred_rgb = self.model(x_rgb)
                    loss     = self.loss_fn(pred_rgb, y)
                main_loss = loss
                c_loss    = torch.tensor(0.0)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            lm_count     += 1
            loss_mean     = 1/lm_count * loss.item()      + (1 - 1/lm_count) * loss_mean
            main_mean     = 1/lm_count * main_loss.item() + (1 - 1/lm_count) * main_mean
            consist_mean  = 1/lm_count * c_loss.item()    + (1 - 1/lm_count) * consist_mean

            pbar.set_postfix(
                loss=f"{loss_mean:.4f}",
                main=f"{main_mean:.4f}",
                c=f"{consist_mean:.4f}",
            )

        return loss_mean, main_mean, consist_mean

    # ----------------------------------------------------------
    #  Val epoch — IoU считается на усреднении трёх пространств
    # ----------------------------------------------------------

    def _val_epoch(self, loader):
        self.model.eval()
        val_loss = 0
        val_iou  = 0
        lm_count = 0

        with torch.no_grad():
            for x_rgb, y in tqdm(loader, leave=False, desc="val"):
                x_rgb = x_rgb.to(self.device)
                y     = y.to(self.device)

                # Loss — только RGB (быстро и достаточно для scheduler)
                with torch.amp.autocast("cuda"):
                    pred_rgb = self.model(x_rgb)
                    loss     = self.loss_fn(pred_rgb, y)

                # IoU — усреднение трёх пространств (честная оценка)
                _, _, _, pred_avg = self._multi_view_predict(x_rgb)

                lm_count += 1
                val_loss  = 1/lm_count * loss.item()            + (1 - 1/lm_count) * val_loss
                val_iou   = 1/lm_count * iou_score(pred_avg, y) + (1 - 1/lm_count) * val_iou

        return val_loss, val_iou

    # ----------------------------------------------------------
    #  Основной цикл
    # ----------------------------------------------------------

    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(1, epochs + 1):
            loss, main, consist = self._train_epoch(train_loader, epoch, epochs)
            val_loss, val_iou   = self._val_epoch(val_loader)

            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch}/{epochs}]  "
                f"loss={loss:.4f}  main={main:.4f}  consist={consist:.4f}  "
                f"val={val_loss:.4f}  IoU={val_iou:.4f}  lr={lr:.6f}"
            )

            if val_iou > self.best_iou:
                self.best_iou = val_iou
                path = os.path.join(self.checkpoint_dir, "best_model.pth")
                torch.save(self.model.state_dict(), path)
                print(f"  ✓ лучшая модель сохранена (IoU={self.best_iou:.4f})")

        print(f"\nГотово. Лучший IoU: {self.best_iou:.4f}")
        return self.best_iou
