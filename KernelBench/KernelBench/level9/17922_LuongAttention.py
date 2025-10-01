import torch
import torch.nn.functional as F
from torch import nn


class LuongAttention(nn.Module):
    """
    Luong Attention from Effective Approaches to Attention-based Neural Machine Translation
    https://arxiv.org/pdf/1508.04025.pdf
    """

    def __init__(self, attention_dim):
        super(LuongAttention, self).__init__()
        self.W = nn.Linear(attention_dim, attention_dim, bias=False)

    def score(self, decoder_hidden, encoder_out):
        encoder_out = self.W(encoder_out)
        encoder_out = encoder_out.permute(1, 0, 2)
        return encoder_out @ decoder_hidden.permute(1, 2, 0)

    def forward(self, decoder_hidden, encoder_out):
        energies = self.score(decoder_hidden, encoder_out)
        mask = F.softmax(energies, dim=1)
        context = encoder_out.permute(1, 2, 0) @ mask
        context = context.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)
        return context, mask


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'attention_dim': 4}]
