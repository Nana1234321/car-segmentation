import cv2
import numpy as np
import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2
        score = 2 * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        return 1 - score.sum() / num


class WeightedBCELoss(nn.Module):
    """
    BCE где пиксели на границе маски весят в boundary_weight раз больше.
    Идея из решения победителей Carvana — граница машины критичнее всего.
    """
    def __init__(self, boundary_weight=3.0):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.kernel = np.ones((5, 5), np.uint8)

    def forward(self, logits, targets):
        weights = torch.ones_like(targets)
        targets_np = targets.cpu().numpy().astype(np.uint8)

        for i in range(targets_np.shape[0]):
            mask = targets_np[i, 0]
            dilated  = cv2.dilate(mask, self.kernel, iterations=2)
            eroded   = cv2.erode(mask,  self.kernel, iterations=2)
            boundary = (dilated - eroded).astype(np.float32)
            weights[i, 0] = torch.from_numpy(
                1.0 + (self.boundary_weight - 1.0) * boundary
            ).to(targets.device)

        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=weights, reduction="mean"
        )


class CombinedLoss(nn.Module):
    """WeightedBCE + SoftDice — стандартная комбинация для сегментации"""
    def __init__(self, boundary_weight=3.0):
        super().__init__()
        self.bce  = WeightedBCELoss(boundary_weight)
        self.dice = SoftDiceLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)
