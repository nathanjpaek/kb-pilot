import torch


class ConvLayer(torch.nn.Module):
    """
        A small wrapper around nn.Conv2d, so as to make the code cleaner and allow for experimentation with padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels,
            kernel_size, stride, padding=kernel_size // 2, padding_mode=
            'reflect')

    def forward(self, x):
        return self.conv2d(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'stride': 1}]
