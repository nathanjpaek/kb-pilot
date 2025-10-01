from torch.nn import Module
import torch
import torch.nn as nn
from torch.nn.modules.module import Module


class UpdateFunc(Module):
    """Implements a Message function"""

    def __init__(self, sa_dim, n_agents, hidden_size):
        super(UpdateFunc, self).__init__()
        self.fv = nn.Linear(hidden_size + sa_dim, hidden_size)
        self.input_dim = hidden_size + sa_dim
        self.output_dim = hidden_size
        self.n_agents = n_agents

    def forward(self, input_feature, x, extended_adj):
        """
          :param input_feature: [batch_size, n_agent ** 2, self.sa_dim] tensor
          :param x: [batch_size, n_agent, self.sa_dim] tensor
          :param extended_adj: [n_agent, n_agent ** 2] tensor
          :return v: [batch_size, n_agent, hidden_size] tensor
        """
        agg = torch.matmul(extended_adj, input_feature)
        x = torch.cat((agg, x), dim=2)
        v = self.fv(x)
        return v

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim
            ) + ' -> ' + str(self.output_dim) + ')'


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'sa_dim': 4, 'n_agents': 4, 'hidden_size': 4}]
