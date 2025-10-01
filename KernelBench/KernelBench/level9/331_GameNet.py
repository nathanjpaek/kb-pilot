import torch
import torch.utils.data


class GameNet(torch.nn.Module):

    def __init__(self):
        super(GameNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, (3, 3), stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 1, (3, 3), stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.ReLU()(self.conv1(x))
        x = torch.nn.ReLU()(self.conv2(x))
        x = self.conv3(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
