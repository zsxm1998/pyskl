import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS, build_loss


@HEADS.register_module()
class EnergyEstimateHead(nn.Module):
    """ A energy expenditure regression head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 in_channels,
                 loss_func=dict(type='MSELoss'),
                 dropout=0.2,
                 init_std=0.01,
                 mode='3D'):
        super().__init__()
        self.loss_func = build_loss(loss_func)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        assert mode in ['3D', 'GCN', '2D']
        self.mode = mode

        self.in_c = in_channels
        self.fc = nn.Sequential(nn.Linear(self.in_c, 1), nn.ReLU())

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)

                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)

        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)

        eee = self.fc(x)
        return eee
    
    def loss(self, eee, label):
        return self.loss_func(eee, label)