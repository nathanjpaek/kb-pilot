import torch
import torch.nn.functional as F


class AdapterModule(torch.nn.Module):

    def __init__(self, d_in, adapter_size):
        super().__init__()
        self.project_down = torch.nn.Linear(d_in, adapter_size)
        self.project_up = torch.nn.Linear(adapter_size, d_in)

    def forward(self, x):
        i1 = self.project_down(x)
        i2 = F.relu(i1)
        i3 = self.project_up(i2)
        f = i3 + x
        return f


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_in': 4, 'adapter_size': 4}]
