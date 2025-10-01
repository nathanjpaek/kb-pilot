import torch
import torch.nn as nn
import torch.nn.parallel


class ECALayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1
            ) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _b, _c, _h, _w = x.size()
        y = torch.mean(x, dim=(2, 3), keepdim=True)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2
            ).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel': 4}]
