import torch
import torch.nn as nn
import torch.utils.data


class RewardModelNetwork(nn.Module):

    def __init__(self, input_size: 'int', hidden_size: 'int', output_size:
        'int') ->None:
        super(RewardModelNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
        self.a1 = nn.Tanh()
        self.a2 = nn.Sigmoid()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        out = x
        out = self.l1(out)
        out = self.a1(out)
        out = self.l2(out)
        out = self.a2(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}]
