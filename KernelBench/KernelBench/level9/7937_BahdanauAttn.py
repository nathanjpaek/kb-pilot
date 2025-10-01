import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttn(nn.Module):

    def __init__(self, context_size, hidden_size):
        super(BahdanauAttn, self).__init__()
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.attn_h = nn.Linear(self.hidden_size, self.context_size, bias=False
            )
        self.attn_e = nn.Linear(self.context_size, self.context_size, bias=
            False)
        self.v = nn.Parameter(torch.rand(self.context_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, ctx_mask=None):
        """
        :param hidden: 
            previous hidden state of the decoder, in shape (1,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (S,B,C)
        :return
            attention energies in shape (B,S)
        """
        max_len = encoder_outputs.size(0)
        encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(H, encoder_outputs)
        if ctx_mask is not None:
            self.mask = (1 - ctx_mask.transpose(0, 1).data).byte()
            attn_energies.data.masked_fill_(self.mask, -float('inf'))
        return self.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn_h(hidden) + self.attn_e(encoder_outputs))
        energy = energy.transpose(2, 1)
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'context_size': 4, 'hidden_size': 4}]
