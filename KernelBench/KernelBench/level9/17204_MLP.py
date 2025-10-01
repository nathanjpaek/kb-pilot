import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import init


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, act=nn.ReLU(),
        normalize_input=True):
        super(MLP, self).__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, output_dim)
        self.act = act
        self.normalize_input = normalize_input
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn
                    .init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        if self.normalize_input:
            x = (x - torch.mean(x, dim=0)) / torch.std(x, dim=0)
        x = self.act(self.linear_1(x))
        return self.linear_2(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4}]
