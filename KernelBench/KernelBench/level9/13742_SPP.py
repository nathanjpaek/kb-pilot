import torch
import torch.nn as nn


class SPP(nn.Module):
    """
    Spatial pyramid pooling layer used in YOLOv3-SPP
    """

    def __init__(self, kernels=[5, 9, 13]):
        super(SPP, self).__init__()
        self.maxpool_layers = nn.ModuleList([nn.MaxPool2d(kernel_size=
            kernel, stride=1, padding=kernel // 2) for kernel in kernels])

    def forward(self, x):
        out = torch.cat([x] + [layer(x) for layer in self.maxpool_layers],
            dim=1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
