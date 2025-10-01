import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class BilinearRanking(nn.Module):

    def __init__(self, n_classes: 'int'=2, emb_size: 'int'=768, block_size:
        'int'=8):
        super().__init__()
        self.n_classes = n_classes
        self.emb_size = emb_size
        self.block_size = block_size
        self.bilinear = nn.Linear(self.emb_size * self.block_size, self.
            n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text1: 'Tensor', text2: 'Tensor'):
        b1 = text1.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = text2.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size *
            self.block_size)
        logits = self.bilinear(bl)
        softmax_logits = self.softmax(logits)
        log_softmax = F.log_softmax(logits, dim=-1)
        return softmax_logits, log_softmax


def get_inputs():
    return [torch.rand([4, 96, 8]), torch.rand([4, 96, 8])]


def get_init_inputs():
    return [[], {}]
