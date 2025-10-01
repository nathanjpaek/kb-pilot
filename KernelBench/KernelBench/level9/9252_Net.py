import torch
import torch.fft
import torch.nn.functional as torchf


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(4, 2, 3, padding=1)

    def forward(self, x):
        x = torchf.relu(self.conv1(x))
        x = torchf.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def get_inputs():
    return [torch.rand([4, 2, 64, 64])]


def get_init_inputs():
    return [[], {}]
