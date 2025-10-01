import torch
import torch.nn.functional


class AddNet(torch.nn.Module):

    def __init__(self):
        super(AddNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=1, stride=1)

    def forward(self, x, y):
        x = self.conv1(x)
        x = x + 3
        y = self.conv2(y)
        return x - y, y - x, x + y


def get_inputs():
    return [torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
