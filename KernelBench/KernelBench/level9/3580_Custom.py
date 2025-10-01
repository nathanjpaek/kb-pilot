import torch
import torch._utils


class Custom(torch.nn.Module):

    def __init__(self):
        super(Custom, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, 1)
        self.conv1 = torch.nn.Conv2d(3, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        _ = self.conv(x)
        _ = self.conv1(_)
        _ = x + _
        _ = self.relu(_)
        t = self.conv2(_)
        t = self.relu(t)
        r = torch.cat([_, t])
        return r * 100


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
