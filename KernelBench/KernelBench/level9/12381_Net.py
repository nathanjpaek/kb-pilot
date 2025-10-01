import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(3 * 16 * 16, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(-1, 3 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 24, 24])]


def get_init_inputs():
    return [[], {}]
