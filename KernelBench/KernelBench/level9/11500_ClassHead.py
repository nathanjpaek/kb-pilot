import torch
from torch import nn
import torch.cuda


class ClassHead(nn.Module):
    """
    ClassHead

    RetinaFace head for classification branch.

    Args:
        inchannels (`int`):
            number of input channels.
        num_anchors (`int`):
            number of anchors.
    """

    def __init__(self, inchannels: 'int'=512, num_anchors: 'int'=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2,
            kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x: 'torch.FloatTensor'):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


def get_inputs():
    return [torch.rand([4, 512, 64, 64])]


def get_init_inputs():
    return [[], {}]
