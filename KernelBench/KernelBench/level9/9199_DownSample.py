import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.functional
import torch.autograd


class Smooth(nn.Module):
    """
    <a id="smooth"></a>
    ### Smoothing Layer

    This layer blurs each channel
    """

    def __init__(self):
        super().__init__()
        kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: 'torch.Tensor'):
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)
        x = self.pad(x)
        x = F.conv2d(x, self.kernel)
        return x.view(b, c, h, w)


class DownSample(nn.Module):
    """
    <a id="down_sample"></a>
    ### Down-sample

    The down-sample operation [smoothens](#smooth) each feature channel and
     scale $2 	imes$ using bilinear interpolation.
    This is based on the paper
     [Making Convolutional Networks Shift-Invariant Again](https://papers.labml.ai/paper/1904.11486).
    """

    def __init__(self):
        super().__init__()
        self.smooth = Smooth()

    def forward(self, x: 'torch.Tensor'):
        x = self.smooth(x)
        return F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2), mode=
            'bilinear', align_corners=False)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
