import torch
from torch import nn as nn
from torch.nn import functional as F


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.decoder_proj = nn.Linear(hidden_size, hidden_size)
        self.encoder_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.decoder_proj.weight)
        nn.init.xavier_uniform_(self.encoder_proj.weight)

    def forward(self, decoder_hidden, encoder_outputs, encoder_masks=None):
        query = self.decoder_proj(decoder_hidden)
        key = self.encoder_proj(encoder_outputs)
        energy = torch.sum(torch.mul(key, query.unsqueeze(1)), dim=-1)
        if encoder_masks is not None:
            energy.masked_fill_(~encoder_masks, -1 * 10000000.0)
        attn_dists = F.softmax(energy, dim=-1)
        context_vecs = torch.sum(torch.mul(encoder_outputs, attn_dists.
            unsqueeze(2)), dim=1)
        return context_vecs, attn_dists


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
