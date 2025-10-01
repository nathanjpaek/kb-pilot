import torch


class _MixPool2d(torch.nn.Module):

    def __init__(self, kernel_size, stride, padding=0, ceil_mode=False):
        super(_MixPool2d, self).__init__()
        self.max_pool = torch.nn.MaxPool2d(kernel_size, stride, padding,
            ceil_mode=ceil_mode)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size, stride, padding,
            ceil_mode=ceil_mode)

    def forward(self, input):
        return self.max_pool(input) + self.avg_pool(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4, 'stride': 1}]
