import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class EEMMRecognizer3D(BaseRecognizer):

    def forward_train(self, imgs, heart_rate, weight, height, age, sex, label, **kwargs):
        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        feat = self.extract_feat(imgs)
        cls_score = self.cls_head(feat, heart_rate, weight, height, age, sex)

        gt_label = label.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_label, **kwargs)

        return loss_cls

    def forward_test(self, imgs, heart_rate, weight, height, age, sex, **kwargs):
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        feat = self.extract_feat(imgs)

        # should have cls_head if not extracting features
        assert self.with_cls_head
        cls_score = self.cls_head(feat, heart_rate, weight, height, age, sex)
        cls_score = cls_score.reshape(batches, num_segs, cls_score.shape[-1])
        cls_score = self.average_clip(cls_score)
        return cls_score.cpu().numpy()
    
    def forward(self, imgs, heart_rate, weight, height, age, sex, label=None, return_loss=True, **kwargs):
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(imgs, heart_rate, weight, height, age, sex, label, **kwargs)

        return self.forward_test(imgs, heart_rate, weight, height, age, sex, **kwargs)
