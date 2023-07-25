# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import random

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
        if label.ndim == 0:
            label = label.unsqueeze(-1).unsqueeze(-1)
        if label.ndim == 1:
            label = label.unsqueeze(-1)
        assert eee.shape == label.shape, f'eee: {eee.shape}, label: {label.shape}'
        loss = (torch.abs(eee-label) / label.clip(min=1e-6)).mean()
        return dict(percentage = loss)

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
        if label.ndim == 0:
            label = label.unsqueeze(-1).unsqueeze(-1)
        if label.ndim == 1:
            label = label.unsqueeze(-1)
        assert eee.shape == label.shape, f'eee: {eee.shape}, label: {label.shape}'
        loss = F.mse_loss(eee, label)
        return dict(mse_loss = loss)
    
@LOSSES.register_module()
class L1Loss(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)

    def _forward(self, eee, label):
        """Forward function.

        Args:
            eee (torch.Tensor): The energy expenditure estimation value predicted by model.
            label (torch.Tensor): The ground truth label.

        Returns:
            torch.Tensor: The returned L1Loss loss.
        """
        if label.ndim == 0:
            label = label.unsqueeze(-1).unsqueeze(-1)
        if label.ndim == 1:
            label = label.unsqueeze(-1)
        assert eee.shape == label.shape, f'eee: {eee.shape}, label: {label.shape}'
        loss = F.l1_loss(eee, label)
        return dict(l1_loss = loss)
    
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
        if label.ndim == 0:
            label = label.unsqueeze(-1)
        cls_idx = torch.argmax(cls_score, dim=1)
        assert cls_idx.shape == label.shape, f'cls_idx: {cls_idx.shape}, label: {label.shape}'
        cls_idx = cls_idx * self.bin
        label = label * self.bin
        loss = (torch.abs(cls_idx-label) / label.clip(min=1e-6)).mean()
        return dict(bin_percentage = loss)
    
# @LOSSES.register_module()
# class BinCrossEntropy(BaseWeightedLoss):
#     def __init__(self, size=5, sigma=0.6, loss_weight=1.0):
#         super().__init__(loss_weight=loss_weight)
#         kernel = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
#         kernel = torch.exp(-kernel.pow(2) / (2 * sigma**2))
#         self.kernel = kernel / kernel.sum()
#         self.half_size = (size - 1) // 2

#     def _forward(self, cls_score, label):
#         """Forward function.

#         Args:
#             cls_score (torch.Tensor): The class score.
#             label (torch.Tensor): The ground truth label.

#         Returns:
#             torch.Tensor: The returned BinCrossEntropy loss.
#         """
#         if label.ndim == 0:
#             label = label.unsqueeze(-1)
#         smooth_label = torch.zeros_like(cls_score)
#         for i, clsidx in enumerate(label):
#             ks = max(-clsidx+self.half_size, 0)
#             ke = self.kernel.size(0) - max(clsidx + self.half_size - cls_score.size(1) + 1, 0)
#             smooth_label[i, max(0, clsidx-self.half_size): clsidx+self.half_size+1] = self.kernel[ks:ke]
        
#         loss = F.cross_entropy(cls_score, smooth_label)
#         return dict(bin_ce_loss = loss)
@LOSSES.register_module()
class BinCrossEntropy(BaseWeightedLoss):
    def __init__(self, temperature=0.1, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)
        self.temperature = temperature

    def _forward(self, cls_score, label):
        if label.ndim == 0:
            label = label.unsqueeze(-1)
        smooth_label = torch.arange(cls_score.size(1)).repeat(cls_score.size(0), 1).to(label)
        smooth_label -= label.unsqueeze(-1)
        sigma = torch.tensor(cls_score.size(1)/16).sqrt()
        smooth_label = torch.exp(-smooth_label.pow(2) / (2 * sigma**2))
        smooth_label = torch.softmax(smooth_label/self.temperature, dim=1)
        loss = F.cross_entropy(cls_score, smooth_label)
        return dict(bin_ce_loss = loss)

class GradientPool():
    def __init__(self, label, total_epochs, num_classes, err_range):
        self.pool = torch.zeros(num_classes, dtype=torch.float64)

        self.origin = torch.zeros(num_classes)
        self.origin[label] = 1

        self.random = torch.zeros(num_classes)
        ksize = 2 * err_range + 1
        #self.random[max(0, label-err_range): label+err_range+1] = 1 / (ksize)
        kernel = torch.arange(ksize, dtype=torch.float32) - err_range
        kernel = torch.exp(-kernel.pow(2) / (2 * (ksize)**2))
        ks = max(-label+err_range, 0)
        ke = ksize - max(label + err_range + 1 - num_classes, 0)
        self.random[max(0, label-err_range): label+err_range+1] = (kernel / kernel.sum())[ks:ke]

        self.total_epochs = total_epochs
        self.num_classes = num_classes
        self.epoch = 0

    @torch.no_grad()
    def get_label(self, result: torch.Tensor, lambda1: float=2., lambda2: float=1.):
        assert result.ndim == 1, f'result should be a 1-dim tensor, but got shape: {result.shape}'
        prob, pred_index = result.detach().cpu().softmax(dim=0).max(dim=0)
        pred = torch.zeros_like(self.origin)
        pred[pred_index] = 1

        self.pool += prob*(1-self.epoch/self.total_epochs)*self.origin \
                   + prob*lambda1*self.epoch/self.total_epochs*pred \
                   + lambda2*self.epoch/self.total_epochs*self.random

        label = torch.tensor(random.choices(range(self.num_classes), weights=self.pool)[0], dtype=torch.int64)
        self.epoch += 1
        return label


@LOSSES.register_module()
class GSSLoss(BaseWeightedLoss):
    def __init__(self, 
                 total_epochs, 
                 num_classes, 
                 err_range,
                 lambda1=2.0,
                 lambda2=1.0,
                 temperature = 0.1,
                 loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)
        self.bin_ce = BinCrossEntropy(temperature=temperature, loss_weight=1.0)
        self.total_epoch = total_epochs
        self.num_classes = num_classes
        self.err_range = err_range
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.gsspool = dict()

    def _forward(self, cls_scores, labels, keys):
        if labels.ndim == 0:
            labels = labels.unsqueeze(-1)
        
        gsslabels = []
        for score, label, key in zip(cls_scores, labels, keys):
            pool = self.gsspool.setdefault(key, GradientPool(label, self.total_epoch, self.num_classes, self.err_range))
            gsslabels.append(pool.get_label(score, self.lambda1, self.lambda2))
        gsslabels = torch.stack(gsslabels).to(labels)
        
        loss = self.bin_ce(cls_scores, gsslabels)
        loss['gss_bin_ce_loss'] = loss.pop('bin_ce_loss')
        return loss