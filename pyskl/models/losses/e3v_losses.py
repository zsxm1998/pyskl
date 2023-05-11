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
            torch.Tensor: The returned CrossEntropy loss.
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
            torch.Tensor: The returned CrossEntropy loss.
        """
        if len(label.shape) == 0:
            label = label.unsqueeze(-1).unsqueeze(-1)
        if len(label.shape) == 1:
            label = label.unsqueeze(-1)
        assert eee.shape == label.shape, f'eee: {eee.shape}, label: {label.shape}'
        loss = F.mse_loss(eee, label)
        return dict(loss = loss)
