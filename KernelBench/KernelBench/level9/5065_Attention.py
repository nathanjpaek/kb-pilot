import torch
import torch as th
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention_dim = attention_dim
        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)
        self.A = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        u_hs = self.U(features)
        w_ah = self.W(hidden_state)
        combined_states = th.tanh(u_hs + w_ah.unsqueeze(1))
        attention_scores = self.A(combined_states)
        attention_scores = attention_scores.squeeze(2)
        alpha = F.softmax(attention_scores, dim=1)
        attention_weights = features * alpha.unsqueeze(2)
        attention_weights = attention_weights.sum(dim=1)
        return attention_weights


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'encoder_dim': 4, 'decoder_dim': 4, 'attention_dim': 4}]
