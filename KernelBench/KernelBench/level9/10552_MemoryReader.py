import torch
import torch.nn as nn


class MemoryReader(nn.Module):

    def __init__(self, state_size, memory_size, h_size, device):
        super(MemoryReader, self).__init__()
        self.device = device
        self.state_size = state_size
        self.memory_size = memory_size
        self.h_size = h_size
        self.fc_h = nn.Linear(state_size, h_size)
        self.fc_k = nn.Linear(state_size + h_size + memory_size, memory_size)

    def forward(self, state, memory):
        h = self.fc_h(state)
        k = self.fc_k(torch.cat((state, h, memory), dim=1)).sigmoid()
        out = memory * k
        return out


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'memory_size': 4, 'h_size': 4, 'device': 0}]
