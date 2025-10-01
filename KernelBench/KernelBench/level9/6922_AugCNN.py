import torch
import torch.nn as nn
from torch.nn import functional as F


def apply_init_(modules):
    """
    Initialize NN modules
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get('padding', 'SAME')

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(0, (out_size - 1) * self.stride[dim] +
            effective_filter_size - input_size)
        additional_padding = int(total_padding % 2 != 0)
        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == 'VALID':
            return F.conv2d(input, self.weight, self.bias, self.stride,
                padding=0, dilation=self.dilation, groups=self.groups)
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])
        return F.conv2d(input, self.weight, self.bias, self.stride, padding
            =(padding_rows // 2, padding_cols // 2), dilation=self.dilation,
            groups=self.groups)


class AugCNN(nn.Module):
    """
    Convolutional Neural Network used as Augmentation
    """

    def __init__(self):
        super(AugCNN, self).__init__()
        self.aug = Conv2d_tf(3, 3, kernel_size=3)
        apply_init_(self.modules())
        self.train()

    def forward(self, obs):
        return self.aug(obs)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
