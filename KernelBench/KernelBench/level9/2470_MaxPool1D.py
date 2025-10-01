import torch


class MaxPool1D(torch.nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return torch.nn.functional.max_pool1d(x, self.kernel_size, stride=
            self.stride, padding=self.padding, ceil_mode=self.ceil_mode)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
