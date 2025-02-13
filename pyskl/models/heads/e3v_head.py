import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS, build_loss


@HEADS.register_module()
class EnergyEstimateHead(nn.Module):
    """ A energy expenditure regression head.

    Args:
        in_channels (int): Number of channels in input feature.
        loss_func (dict): Config for building loss. Default: dict(type='MSELoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 in_channels,
                 loss_func=dict(type='MSELoss'),
                 dropout=0.0,
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
        # self.fc = nn.Linear(self.in_c, 1)
        self.fc = nn.Sequential(
            nn.Linear(self.in_c, self.in_c),
            nn.Tanh(),
            nn.Linear(self.in_c, 1)
        )
        self.relu = nn.ReLU()

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

        assert x.shape[1] == self.in_c, f'in_channels:{self.in_c}, x.shape:{x.shape}'
        if self.dropout is not None:
            x = self.dropout(x)

        eee = self.fc(x)
        if not self.training:
            eee = self.relu(eee)
        return eee
    
    def loss(self, eee, label):
        return self.loss_func(eee, label)


@HEADS.register_module()
class EEOrdinalHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_func,
                 dropout=0.0,
                 init_std=0.01,
                 mode='3D',
                 **kwargs):
        super().__init__()
        self.gss_flag = loss_func['type'] == 'GSSLoss'
        self.loss_func = build_loss(loss_func)
        self.loss_percentage = build_loss(dict(type='BinPercentageLoss'))

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        assert mode in ['3D', 'GCN', '2D']
        self.mode = mode

        self.in_c = in_channels
        #self.fc_cls = nn.Linear(self.in_c, num_classes)
        self.fc_cls = nn.Sequential(
            nn.Linear(self.in_c, self.in_c),
            nn.Tanh(),
            nn.Linear(self.in_c, num_classes)
        )

    def init_weights(self):
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
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

        cls_score = self.fc_cls(x)
        return cls_score
    
    def loss(self, cls_score, label, id_num=None):
        losses = {}
        if self.gss_flag:
            losses.update(self.loss_func(cls_score, label, keys=id_num))
        else:
            losses.update(self.loss_func(cls_score, label))
        losses.update(self.loss_percentage(cls_score, label))
        return losses

@HEADS.register_module()
class MMEnergyEstimateHead(nn.Module):
    """ A multi modality energy expenditure regression head.

    Args:
        in_channels (int): Number of channels in input feature.
        loss_func (dict): Config for building loss. Default: dict(type='MSELoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 in_channels,
                 loss_func=dict(type='MSELoss'),
                 dropout=0.0,
                 init_std=0.01,
                 mode='3D'):
        super().__init__()
        self.loss_func = build_loss(loss_func)
        self.loss_percentage = build_loss(dict(type='EEPercentageLoss'))

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        assert mode in ['3D', 'GCN', '2D']
        self.mode = mode

        self.in_c = in_channels
        self.fc = nn.Linear(in_channels*2, 1)
        self.relu = nn.ReLU()

        self.mm_branch = nn.Sequential(
            nn.Linear(5, in_channels), #5是因为heart_rate, weight, height, age, sex
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
        )

    def init_weights(self):
        normal_init(self.fc, std=self.init_std)

    def forward(self, x, heart_rate, weight, height, age, sex):
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

        assert x.shape[1] == self.in_c, f'in_channels:{self.in_c}, x.shape:{x.shape}'
        if self.dropout is not None:
            x = self.dropout(x)

        heart_rate = heart_rate.to(x.dtype)
        weight = weight.to(x.dtype)
        height = height.to(x.dtype)
        age = age.to(x.dtype)
        sex = sex.to(x.dtype)
        mmfea = torch.cat([heart_rate, weight, height, age, sex], dim=1)
        mmfea = self.mm_branch(mmfea)
        if self.dropout is not None:
            mmfea = self.dropout(mmfea)
        if x.size(0) != mmfea.size(0):
            assert x.size(0) % mmfea.size(0) == 0
            rep_num = x.size(0) // mmfea.size(0)
            mmfea = mmfea.repeat_interleave(rep_num, dim=0)
        x = torch.cat([x, mmfea], dim=1)

        eee = self.fc(x)
        if not self.training:
            eee = self.relu(eee)
        return eee
    
    def loss(self, eee, label):
        losses = {}
        losses.update(self.loss_func(eee, label))
        losses.update(self.loss_percentage(eee, label))
        return losses