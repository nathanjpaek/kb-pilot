import torch
import torch.nn as nn
import torch.nn.functional as F


class Aggregator(nn.Module):

    def __init__(self, hidden_dim, num_node):
        super(Aggregator, self).__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(num_node, 1)

    def forward(self, inputs):
        """

        :param inputs: [B,T,N,D]
        [B,T,N,D] ,[T,N,D]->[B,1,N,D]
        :return: [B,1,N,D]
        """
        q = F.tanh(self.W_q(inputs))
        k = F.tanh(self.W_k(inputs)).transpose(-1, -2)
        attn = torch.einsum('...nd,...bc->...nc', q, k)
        attn = self.fc(attn)
        attn = F.softmax(attn, dim=1)
        ret = torch.einsum('bsnd,bsnl->blnd', inputs, attn)
        return ret


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4, 'num_node': 4}]
