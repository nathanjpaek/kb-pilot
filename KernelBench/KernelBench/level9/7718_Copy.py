import torch
from torch import nn


class Copy(nn.Module):

    def __init__(self, hidden_size, copy_weight=1.0):
        super().__init__()
        self.Wcopy = nn.Linear(hidden_size, hidden_size)
        self.copy_weight = copy_weight

    def forward(self, enc_out_hs, dec_hs):
        """
        get unnormalized copy score
        :param enc_out_hs: [B, Tenc,  H]
        :param dec_hs: [B, Tdec, H]   testing: Tdec=1
        :return: raw_cp_score of each position, size [B, Tdec, Tenc]
        """
        raw_cp_score = torch.tanh(self.Wcopy(enc_out_hs))
        raw_cp_score = torch.einsum('beh,bdh->bde', raw_cp_score, dec_hs)
        return raw_cp_score * self.copy_weight


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
