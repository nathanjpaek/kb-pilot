import torch
import torch.nn as nn
import torch.nn.functional as f


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = f.relu(self.fc1(x.view(-1, 28 * 28)))
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
