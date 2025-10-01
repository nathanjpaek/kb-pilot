import torch
import torch.nn.functional


class ReuseLayerNet(torch.nn.Module):

    def __init__(self):
        super(ReuseLayerNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.identity = torch.nn.Identity()

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.identity(x)
        x = self.conv1(x)
        x = self.identity(x)
        x = self.conv1(x)
        x = self.identity(x)
        x = self.conv2(x)
        x = self.identity(x)
        x = self.conv2(x)
        x = self.identity(x)
        x = self.conv2(x)
        x = self.identity(x)
        y = self.conv2(y)
        y = self.identity(y)
        y = self.conv2(y)
        y = self.identity(y)
        y = self.conv1(y)
        y = self.identity(y)
        y = self.conv1(y)
        y = self.identity(y)
        y = self.conv1(y)
        return x - y, y - x


def get_inputs():
    return [torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
