import torch


class TorchFlattenNet(torch.nn.Module):

    def __init__(self):
        super(TorchFlattenNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        return torch.flatten(x, 1)


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
