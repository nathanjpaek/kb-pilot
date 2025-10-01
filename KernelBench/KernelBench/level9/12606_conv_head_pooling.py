import torch
import torch.nn as nn
import torch.utils.data


class conv_head_pooling(nn.Module):

    def __init__(self, in_feature, out_feature, stride, conv_type,
        padding_mode='zeros', dilation=1):
        super(conv_head_pooling, self).__init__()
        if conv_type == 'depthwise':
            _groups = in_feature
        else:
            _groups = 1
        None
        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=3,
            padding=dilation, dilation=dilation, stride=stride,
            padding_mode=padding_mode, groups=_groups)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):
        x = self.conv(x)
        cls_token = self.fc(cls_token)
        return x, cls_token


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_feature': 4, 'out_feature': 4, 'stride': 1,
        'conv_type': 4}]
