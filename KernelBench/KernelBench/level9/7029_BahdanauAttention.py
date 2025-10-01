import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F


class BahdanauAttention(nn.Module):

    def __init__(self, dec_dim: 'int', enc_dim: 'int', num_hiddens: 'int'):
        super().__init__()
        self.W1 = nn.Linear(dec_dim, num_hiddens, bias=False)
        self.W2 = nn.Linear(enc_dim, num_hiddens, bias=False)
        self.v = nn.Linear(num_hiddens, 1, False)

    def forward(self, query: 'Tensor', value: 'Tensor') ->Tuple[Tensor, Tensor
        ]:
        """
        Args:
            value (Tensor(batch size, seq_len, encoder hidden dimension): the hidden_state of tokens in encoder
            query (Tensor(batch size, 1, decoder hidden dimension)): the hidden state of decoder at time step t
        Returns:
            attention_weight (Tensor)
            context_vector (Tensor)
        """
        score = self.v(torch.tanh(self.W1(query) + self.W2(value)))
        attention_weight = F.softmax(score.squeeze(-1), dim=1)
        context_vector = torch.bmm(attention_weight.unsqueeze(1), value
            ).squeeze(1)
        return attention_weight, context_vector


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dec_dim': 4, 'enc_dim': 4, 'num_hiddens': 4}]
