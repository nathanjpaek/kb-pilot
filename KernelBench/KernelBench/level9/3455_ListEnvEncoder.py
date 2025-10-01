import torch
import torch.nn.functional as F
import torch.nn as nn


class ListEnvEncoder(nn.Module):
    """
    Implement an encoder (f_enc) specific to the List environment. It encodes observations e_t into
    vectors s_t of size D = encoding_dim.
    """

    def __init__(self, observation_dim, encoding_dim):
        super(ListEnvEncoder, self).__init__()
        self.l1 = nn.Linear(observation_dim, 100)
        self.l2 = nn.Linear(100, encoding_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = torch.tanh(self.l2(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'observation_dim': 4, 'encoding_dim': 4}]
