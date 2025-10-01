import torch
import torch.nn as nn
from itertools import chain as chain
import torch.utils.data
import torch.hub
import torch.nn.parallel
import torch.optim


class LearnedPositionalEmbedding3(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        self.a_2 = nn.Parameter(torch.ones_like(pe))
        self.b_2 = nn.Parameter(torch.zeros_like(pe))
        pe.require_grad = True
        pe = pe.unsqueeze(0)
        self.pe = nn.Parameter(pe)
        torch.nn.init.normal_(self.pe, std=d_model ** -0.5)

    def forward(self, x):
        return self.a_2 * self.pe[:, :x.size(1)] + self.b_2


class BERTEmbedding4(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, input_dim, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.learnedPosition = LearnedPositionalEmbedding3(d_model=
            input_dim, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        x = self.learnedPosition(sequence) + sequence
        return self.dropout(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'max_len': 4}]
