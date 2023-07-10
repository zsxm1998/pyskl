# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class EEPercentageLoss(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)

    def _forward(self, eee, label):
        """Forward function.

        Args:
            eee (torch.Tensor): The energy expenditure estimation value predicted by model.
            label (torch.Tensor): The ground truth label.

        Returns:
            torch.Tensor: The returned EEPercentageLoss loss.
        """
        if len(label.shape) == 0:
            label = label.unsqueeze(-1).unsqueeze(-1)
        if len(label.shape) == 1:
            label = label.unsqueeze(-1)
        assert eee.shape == label.shape, f'eee: {eee.shape}, label: {label.shape}'
        loss = (torch.abs(eee-label) / label.clip(min=1e-6)).mean()
        return dict(loss = loss)

@LOSSES.register_module()
class MSELoss(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)

    def _forward(self, eee, label):
        """Forward function.

        Args:
            eee (torch.Tensor): The energy expenditure estimation value predicted by model.
            label (torch.Tensor): The ground truth label.

        Returns:
            torch.Tensor: The returned MSELoss loss.
        """
        if len(label.shape) == 0:
            label = label.unsqueeze(-1).unsqueeze(-1)
        if len(label.shape) == 1:
            label = label.unsqueeze(-1)
        assert eee.shape == label.shape, f'eee: {eee.shape}, label: {label.shape}'
        loss = F.mse_loss(eee, label)
        return dict(loss = loss)
    
@LOSSES.register_module()
class BinCrossEntropy(BaseWeightedLoss):
    def __init__(self, size=5, sigma=0.6, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)
        kernel = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        kernel = torch.exp(-kernel.pow(2) / (2 * sigma**2))
        self.kernel = kernel / kernel.sum()
        self.half_size = (size - 1) / 2

    def _forward(self, cls_score, label):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.

        Returns:
            torch.Tensor: The returned BinCrossEntropy loss.
        """
        if len(label.shape) == 0:
            label = label.unsqueeze(-1)
        smooth_label = torch.zeros_like(cls_score)
        for i, clsidx in enumerate(label):
            ks = max(-clsidx+self.half_size, 0)
            ke = self.kernel.size(0) - max(clsidx + self.half_size - cls_score.size(1) + 1, 0)
            smooth_label[i, max(0, clsidx-self.half_size):] = self.kernel[ks:ke]
        
        loss = F.cross_entropy(cls_score, smooth_label)
        return dict(loss = loss)

@LOSSES.register_module()
class BinPercentageLoss(BaseWeightedLoss):
    def __init__(self, bin=0.1, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)
        self.bin = bin

    def _forward(self, cls_score, label):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.

        Returns:
            torch.Tensor: The returned BinPercentageLoss loss.
        """
        if len(label.shape) == 0:
            label = label.unsqueeze(-1)
        cls_idx = torch.argmax(cls_score, dim=1)
        assert cls_idx.shape == label.shape, f'cls_idx: {cls_idx.shape}, label: {label.shape}'
        cls_idx = cls_idx * self.bin
        label = label * self.bin
        loss = (torch.abs(cls_idx-label) / label.clip(min=1e-6)).mean()
        return dict(loss = loss)