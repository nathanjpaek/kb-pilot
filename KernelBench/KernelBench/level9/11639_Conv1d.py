import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding='same'):
        """
        inputs: [N, T, C_in]
        outputs: [N, T, C_out]
        """
        super().__init__()
        if padding == 'same':
            left = (kernel_size - 1) // 2
            right = kernel_size - 1 - left
            self.pad = left, right
        else:
            self.pad = 0, 0
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)

    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1, 2)
        inputs = F.pad(inputs, self.pad)
        out = self.conv1d(inputs)
        out = torch.transpose(out, 1, 2)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
