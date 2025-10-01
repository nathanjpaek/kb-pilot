import torch
from torch import nn


class Attention(nn.Module):

    def __init__(self, feature_dim, max_seq_len=70):
        super().__init__()
        self.attention_fc = nn.Linear(feature_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1, max_seq_len, 1,
            requires_grad=True))

    def forward(self, rnn_output):
        """
        forward attention scores and attended vectors
        :param rnn_output: (#batch,#seq_len,#feature)
        :return: attended_outputs (#batch,#feature)
        """
        attention_weights = self.attention_fc(rnn_output)
        seq_len = rnn_output.size(1)
        attention_weights = self.bias[:, :seq_len, :] + attention_weights
        attention_weights = torch.tanh(attention_weights)
        attention_weights = torch.exp(attention_weights)
        attention_weights_sum = torch.sum(attention_weights, dim=1, keepdim
            =True) + 1e-07
        attention_weights = attention_weights / attention_weights_sum
        attended = torch.sum(attention_weights * rnn_output, dim=1)
        return attended


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_dim': 4}]
