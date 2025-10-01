import torch
import torch.nn
import torch.nn.functional as Functional


class MyNeural(torch.nn.Module):

    def __init__(self, columns):
        super(MyNeural, self).__init__()
        self.f1 = torch.nn.Linear(columns, 32)
        self.f2 = torch.nn.Linear(32, 16)
        self.f3 = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = self.f1(x)
        x = Functional.relu(x)
        x = self.f2(x)
        x = Functional.relu(x)
        x = self.f3(x)
        x = Functional.sigmoid(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'columns': 4}]
