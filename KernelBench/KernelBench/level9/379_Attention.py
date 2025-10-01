import math
import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    Attention mechanism (Luong)
    """

    def __init__(self, hidden_size, hidden_size1):
        super(Attention, self).__init__()
        self.W_h = nn.Linear(hidden_size + hidden_size1, hidden_size, bias=
            False)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.epsilon = 1e-10
        nn.init.xavier_normal_(self.W_h.weight)

    def forward(self, encoder_outputs, decoder_hidden, inp_mask):
        seq_len = encoder_outputs.size(1)
        H = decoder_hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.W_h(torch.cat([H, encoder_outputs], 2)))
        energy = energy.transpose(2, 1)
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)
        energy = torch.bmm(v, energy).view(-1, seq_len)
        a = torch.softmax(energy, dim=-1) * inp_mask
        normalization_factor = a.sum(1, keepdim=True)
        a = a / (normalization_factor + self.epsilon)
        a = a.unsqueeze(1)
        context = a.bmm(encoder_outputs)
        return a, context


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'hidden_size1': 4}]
