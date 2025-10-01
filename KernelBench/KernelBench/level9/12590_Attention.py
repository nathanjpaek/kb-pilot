import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, src_size, trg_size):
        super().__init__()
        self.W = nn.Bilinear(src_size, trg_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, trg, attention_mask=None):
        """
        src: [src_size]
        trg: [middle_node, trg_size]
        """
        score = self.W(src.unsqueeze(0).expand(trg.size(0), -1), trg)
        score = self.softmax(score)
        value = torch.mm(score.permute(1, 0), trg)
        return score.squeeze(0), value.squeeze(0)


def get_inputs():
    return [torch.rand([4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'src_size': 4, 'trg_size': 4}]
