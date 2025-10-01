import torch
import torch.nn as nn


class MemoryWriter(nn.Module):

    def __init__(self, state_size, memory_size, device):
        super(MemoryWriter, self).__init__()
        self.device = device
        self.state_size = state_size
        self.memory_size = memory_size
        self.fc_r = nn.Linear(state_size + memory_size, memory_size)
        self.fc_z = nn.Linear(state_size + memory_size, memory_size)
        self.fc_c = nn.Linear(state_size + memory_size, memory_size)

    def forward(self, state, memory):
        r = self.fc_r(torch.cat((state, memory), dim=1)).sigmoid()
        z = self.fc_z(torch.cat((state, memory), dim=1)).sigmoid()
        c = self.fc_c(torch.cat((state, r * memory), dim=1)).tanh()
        out = (1 - z) * memory + z * c
        return out


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'memory_size': 4, 'device': 0}]
