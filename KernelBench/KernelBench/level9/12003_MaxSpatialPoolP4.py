import torch


class MaxSpatialPoolP4(torch.nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.inner = torch.nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        y = x.view(x.size(0), -1, x.size(3), x.size(4))
        y = self.inner(y)
        y = y.view(x.size(0), -1, 4, y.size(2), y.size(3))
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
