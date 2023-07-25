# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.runner import DistEvalHook as BasicDistEvalHook
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr


class DistEvalHook(BasicDistEvalHook):
    greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP@', 'Recall@'
    ]
    less_keys = ['loss']

    def __init__(self, *args, save_best='auto', seg_interval=None, **kwargs):
        super().__init__(*args, save_best=save_best, **kwargs)
        self.seg_interval = seg_interval
        if seg_interval is not None:
            assert isinstance(seg_interval, list)
            for i, tup in enumerate(seg_interval):
                assert isinstance(tup, tuple) and len(tup) == 3 and tup[0] < tup[1]
                if i < len(seg_interval) - 1:
                    assert tup[1] == seg_interval[i + 1][0]
            assert self.by_epoch
        assert self.start is None

    def _find_n(self, runner):
        current = runner.epoch
        for seg in self.seg_interval:
            if current >= seg[0] and current < seg[1]:
                return seg[2]
        return None

    def _should_evaluate(self, runner):
        if self.seg_interval is None:
            return super()._should_evaluate(runner)
        n = self._find_n(runner)
        assert n is not None
        return self.every_n_epochs(runner, n)


def confusion_matrix(y_pred, y_real, normalize=None):
    """Compute confusion matrix.

    Args:
        y_pred (list[int] | np.ndarray[int]): Prediction labels.
        y_real (list[int] | np.ndarray[int]): Ground truth labels.
        normalize (str | None): Normalizes confusion matrix over the true
            (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized. Options are
            "true", "pred", "all", None. Default: None.

    Returns:
        np.ndarray: Confusion matrix.
    """
    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(
            f'y_pred must be list or np.ndarray, but got {type(y_pred)}')
    if not y_pred.dtype == np.int64:
        raise TypeError(
            f'y_pred dtype must be np.int64, but got {y_pred.dtype}')

    if isinstance(y_real, list):
        y_real = np.array(y_real)
    if not isinstance(y_real, np.ndarray):
        raise TypeError(
            f'y_real must be list or np.ndarray, but got {type(y_real)}')
    if not y_real.dtype == np.int64:
        raise TypeError(
            f'y_real dtype must be np.int64, but got {y_real.dtype}')

    label_set = np.unique(np.concatenate((y_pred, y_real)))
    num_labels = len(label_set)
    max_label = label_set[-1]
    label_map = np.zeros(max_label + 1, dtype=np.int64)
    for i, label in enumerate(label_set):
        label_map[label] = i

    y_pred_mapped = label_map[y_pred]
    y_real_mapped = label_map[y_real]

    confusion_mat = np.bincount(
        num_labels * y_real_mapped + y_pred_mapped,
        minlength=num_labels**2).reshape(num_labels, num_labels)

    with np.errstate(all='ignore'):
        if normalize == 'true':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=1, keepdims=True))
        elif normalize == 'pred':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=0, keepdims=True))
        elif normalize == 'all':
            confusion_mat = (confusion_mat / confusion_mat.sum())
        confusion_mat = np.nan_to_num(confusion_mat)

    return confusion_mat


def mean_class_accuracy(scores, labels):
    """Calculate mean class accuracy.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.

    Returns:
        np.ndarray: Mean class accuracy.
    """
    pred = np.argmax(scores, axis=1)
    cf_mat = confusion_matrix(pred, labels).astype(float)

    cls_cnt = cf_mat.sum(axis=1)
    cls_hit = np.diag(cf_mat)

    mean_class_acc = np.mean(
        [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])

    return mean_class_acc


def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res


def mean_average_precision(scores, labels):
    """Mean average precision for multi-label recognition.

    Args:
        scores (list[np.ndarray]): Prediction scores of different classes for
            each sample.
        labels (list[np.ndarray]): Ground truth many-hot vector for each
            sample.

    Returns:
        np.float: The mean average precision.
    """
    results = []
    scores = np.stack(scores).T
    labels = np.stack(labels).T

    for score, label in zip(scores, labels):
        precision, recall, _ = binary_precision_recall_curve(score, label)
        ap = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        results.append(ap)
    results = [x for x in results if not np.isnan(x)]
    if results == []:
        return np.nan
    return np.mean(results)


def binary_precision_recall_curve(y_score, y_true):
    """Calculate the binary precision recall curve at step thresholds.

    Args:
        y_score (np.ndarray): Prediction scores for each class.
            Shape should be (num_classes, ).
        y_true (np.ndarray): Ground truth many-hot vector.
            Shape should be (num_classes, ).

    Returns:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.
        thresholds (np.ndarray): Different thresholds at which precision and
            recall are tested.
    """
    assert isinstance(y_score, np.ndarray)
    assert isinstance(y_true, np.ndarray)
    assert y_score.shape == y_true.shape

    # make y_true a boolean vector
    y_true = (y_true == 1)
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind='mergesort')[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    # There may be ties in values, therefore find the `distinct_value_inds`
    distinct_value_inds = np.where(np.diff(y_score))[0]
    threshold_inds = np.r_[distinct_value_inds, y_true.size - 1]
    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_inds]
    fps = 1 + threshold_inds - tps
    thresholds = y_score[threshold_inds]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def ee_loss(scores, labels, func):
    """Loss of energy expenditure.

    Args:
        scores (list[np.ndarray]): Prediction value of energy.
        
        labels (list[np.ndarray]): Ground truth energy expenditure.

        func (str): "mse" or "percentage"

    Returns:
        np.float: The loss value.
    """
    scores = torch.from_numpy(np.stack(scores))
    labels = torch.from_numpy(np.stack(labels))
    if len(labels.shape) == 1:
        labels = labels.unsqueeze(-1)

    assert scores.shape == labels.shape, f'scores: {scores.shape}, labels: {labels.shape}'
    if func == 'mse':
        loss = F.mse_loss(scores, labels)
    elif func == 'percentage':
        loss = (torch.abs(scores-labels) / labels.clip(min=1e-6)).mean()
    elif func == 'l1':
        loss = F.l1_loss(scores, labels)
    else:
        raise NotImplementedError(f'"func" should be "mse" or "percentage", but got {func}')
    return loss.numpy()


# def bin_cross_entropy(scores, labels, size=5, sigma=0.6):
#     scores = torch.from_numpy(np.stack(scores))
#     labels = torch.from_numpy(np.stack(labels))

#     kernel = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
#     kernel = torch.exp(-kernel.pow(2) / (2 * sigma**2))
#     kernel = kernel / kernel.sum()
#     half_size = (size - 1) // 2

#     smooth_labels = torch.zeros_like(scores)
#     for i, clsidx in enumerate(labels):
#         ks = max(-clsidx+half_size, 0)
#         ke = kernel.size(0) - max(clsidx + half_size - scores.size(1) + 1, 0)
#         smooth_labels[i, max(0, clsidx-half_size): clsidx+half_size+1] = kernel[ks:ke]
    
#     return F.cross_entropy(scores, smooth_labels).numpy()
def bin_cross_entropy(scores, labels, temperature=0.1):
    scores = torch.from_numpy(np.stack(scores))
    labels = torch.from_numpy(np.stack(labels))

    smooth_labels = torch.arange(scores.size(1)).repeat(scores.size(0), 1)
    smooth_labels -= labels.unsqueeze(-1)
    sigma = torch.tensor(scores.size(1)/16).sqrt()
    smooth_labels = torch.exp(-smooth_labels.pow(2) / (2 * sigma**2))
    smooth_labels = torch.softmax(smooth_labels/temperature, dim=1)
    
    return F.cross_entropy(scores, smooth_labels).numpy()


def bin_percentage_lossfunc(scores, labels, bin=0.1):
    scores = torch.from_numpy(np.stack(scores))
    labels = torch.from_numpy(np.stack(labels))

    cls_idx = torch.argmax(scores, dim=1)
    assert cls_idx.shape == labels.shape, f'cls_idx: {cls_idx.shape}, labels: {labels.shape}'
    cls_idx = cls_idx * bin
    labels = labels * bin
    loss = (torch.abs(cls_idx-labels) / labels.clip(min=1e-6)).mean()
    
    return loss.numpy()


def correlation_coefficient(scores, labels, mode, bin=0.1):
    if mode == 'bin':
        scores = np.stack(scores).argmax(axis=1) * bin
        labels = np.stack(labels) * bin
    elif mode == 'regression':
        scores = np.stack(scores).squeeze(axis=1)
        labels = np.stack(labels)
    else:
        raise NotImplementedError(f'"mode" should be "bin" or "regression", but got {mode}')
    pearson, _ = pearsonr(scores, labels)
    spearman, _ = spearmanr(scores, labels)
    return pearson, spearman
