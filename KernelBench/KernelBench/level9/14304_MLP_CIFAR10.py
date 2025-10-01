import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_CIFAR10(nn.Module):

    def __init__(self, save_features=None, bench_model=False):
        super(MLP_CIFAR10, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x0 = F.relu(self.fc1(x.view(-1, 3 * 32 * 32)))
        x1 = F.relu(self.fc2(x0))
        return F.log_softmax(self.fc3(x1), dim=1)


def get_inputs():
    return [torch.rand([4, 3072])]


def get_init_inputs():
    return [[], {}]
