import torch
import torch.nn as nn


def get_operator_from_cfg(operator_cfg):
    operator_cfg_copy = operator_cfg.copy()
    construct_str = 'nn.'
    construct_str += operator_cfg_copy.pop('type') + '('
    for k, v in operator_cfg_copy.items():
        construct_str += k + '=' + str(v) + ','
    construct_str += ')'
    return eval(construct_str)


class FastBlock(nn.Module):

    def __init__(self, num_input_channels, num_block_channels, stride=1,
        downsample=None, activation_cfg=dict(type='ReLU', inplace=True),
        norm_cfg=None):
        super(FastBlock, self).__init__()
        if downsample is not None:
            assert stride == 2
        if norm_cfg is not None:
            assert norm_cfg['type'] in ['BatchNorm2d', 'GroupNorm']
        self._num_input_channel = num_input_channels
        self._num_block_channel = num_block_channels
        self._stride = stride
        self._activation_cfg = activation_cfg
        self._norm_cfg = norm_cfg
        self._downsample = downsample
        self._conv1 = nn.Conv2d(in_channels=self._num_input_channel,
            out_channels=self._num_block_channel, kernel_size=3, stride=
            self._stride, padding=1, bias=True if self._norm_cfg is None else
            False)
        if self._norm_cfg is not None:
            temp_norm_cfg = self._norm_cfg.copy()
            if temp_norm_cfg['type'] == 'BatchNorm2d':
                temp_norm_cfg['num_features'] = self._num_block_channel
            else:
                temp_norm_cfg['num_channels'] = self._num_block_channel
            self._norm1 = get_operator_from_cfg(temp_norm_cfg)
        self._activation = get_operator_from_cfg(self._activation_cfg)
        self._conv2 = nn.Conv2d(in_channels=self._num_block_channel,
            out_channels=self._num_block_channel, kernel_size=1, stride=1,
            padding=0, bias=True if self._norm_cfg is None else False)
        if self._norm_cfg is not None:
            temp_norm_cfg = self._norm_cfg.copy()
            if temp_norm_cfg['type'] == 'BatchNorm2d':
                temp_norm_cfg['num_features'] = self._num_block_channel
            else:
                temp_norm_cfg['num_channels'] = self._num_block_channel
            self._norm2 = get_operator_from_cfg(temp_norm_cfg)
        self._conv3 = nn.Conv2d(in_channels=self._num_block_channel,
            out_channels=self._num_block_channel, kernel_size=3, stride=1,
            padding=1, bias=True if self._norm_cfg is None else False)
        if self._norm_cfg is not None:
            temp_norm_cfg = self._norm_cfg.copy()
            if temp_norm_cfg['type'] == 'BatchNorm2d':
                temp_norm_cfg['num_features'] = self._num_block_channel
            else:
                temp_norm_cfg['num_channels'] = self._num_block_channel
            self._norm3 = get_operator_from_cfg(temp_norm_cfg)

    def forward(self, x):
        identity = x
        out = self._conv1(x)
        if self._norm_cfg is not None:
            out = self._norm1(out)
        out = self._activation(out)
        out = self._conv2(out)
        if self._norm_cfg is not None:
            out = self._norm2(out)
        out = self._activation(out)
        out = self._conv3(out)
        if self._norm_cfg is not None:
            out = self._norm3(out)
        if self._downsample is not None:
            identity = self._downsample(x)
        out += identity
        out = self._activation(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_input_channels': 4, 'num_block_channels': 4}]
