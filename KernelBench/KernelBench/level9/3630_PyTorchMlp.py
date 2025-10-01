import torch
import torch.nn as nn


class PyTorchMlp(nn.Module):

    def __init__(self, n_inputs=4, n_actions=2):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(n_inputs, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)
        self.activ_fn = nn.ReLU()
        self.out_activ = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.activ_fn(self.fc1(x))
        x = self.activ_fn(self.fc2(x))
        x = self.out_activ(self.fc3(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
