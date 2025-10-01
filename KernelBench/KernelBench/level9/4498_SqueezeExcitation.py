import torch
import torch.utils.data


def _make_divisible(width, divisor=8):
    new_width = max(divisor, int(width + divisor / 2) // divisor * divisor)
    if new_width < 0.9 * width:
        new_width += divisor
    return new_width


class SqueezeExcitation(torch.nn.Module):
    """ [https://arxiv.org/abs/1709.01507] """

    def __init__(self, c1):
        super().__init__()
        c2 = _make_divisible(c1 // 4)
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = torch.nn.Conv2d(in_channels=c1, out_channels=c2,
            kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_channels=c2, out_channels=c1,
            kernel_size=1)
        self.relu = torch.nn.ReLU()
        self.hard = torch.nn.Hardsigmoid()

    def _scale(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hard(x)
        return x

    def forward(self, x):
        return x * self._scale(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'c1': 4}]
