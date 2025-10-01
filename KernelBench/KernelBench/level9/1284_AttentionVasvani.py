import torch
from torch import nn


class AttentionVasvani(nn.Module):

    def __init__(self, encoder_dim=128, decoder_dim=128):
        super(AttentionVasvani, self).__init__()

    def forward(self, k, q):
        x = torch.sum(k * q, dim=1, keepdim=True)
        x /= torch.sqrt(torch.norm(k, p=1, dim=1, keepdim=True))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
