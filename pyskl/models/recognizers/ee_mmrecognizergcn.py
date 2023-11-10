import numpy as np
import torch

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class EEMMRecognizerGCN(BaseRecognizer):

    def forward_train(self, keypoint, heart_rate, weight, height, age, sex, label, **kwargs):
        assert self.with_cls_head
        assert keypoint.shape[1] == 1
        keypoint = keypoint[:, 0]

        losses = dict()
        x = self.extract_feat(keypoint)
        cls_score = self.cls_head(x, heart_rate, weight, height, age, sex)
        gt_label = label.squeeze(-1)
        loss = self.cls_head.loss(cls_score, gt_label)
        losses.update(loss)

        return losses

    def forward_test(self, keypoint, heart_rate, weight, height, age, sex, **kwargs):
        assert self.with_cls_head or self.feat_ext
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc, ) + keypoint.shape[2:])

        x = self.extract_feat(keypoint)

        cls_score = self.cls_head(x, heart_rate, weight, height, age, sex)
        cls_score = cls_score.reshape(bs, nc, cls_score.shape[-1])
        # harmless patch
        if 'average_clips' not in self.test_cfg:
            self.test_cfg['average_clips'] = 'score'

        cls_score = self.average_clip(cls_score)
        if isinstance(cls_score, tuple) or isinstance(cls_score, list):
            cls_score = [x.data.cpu().numpy() for x in cls_score]
            return [[x[i] for x in cls_score] for i in range(bs)]

        return cls_score.data.cpu().numpy()

    def forward(self, keypoint, heart_rate, weight, height, age, sex, label=None, return_loss=True, **kwargs):
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(keypoint, heart_rate, weight, height, age, sex, label, **kwargs)

        return self.forward_test(keypoint, heart_rate, weight, height, age, sex, **kwargs)

    def extract_feat(self, keypoint):
        return self.backbone(keypoint)
