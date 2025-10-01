import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GenerationProbabilty(nn.Module):

    def __init__(self, embedding_size, hidden_size, h_star_size):
        """Calculates `p_gen` as described in Pointer-Generator Networks paper."""
        super(GenerationProbabilty, self).__init__()
        self.W_h_star = nn.Linear(h_star_size, 1, bias=False)
        self.W_s = nn.Linear(hidden_size, 1, bias=False)
        self.W_x = nn.Linear(embedding_size, 1, bias=False)
        self.b_attn = nn.Parameter(torch.Tensor(1))
        self.init_parameters()

    def forward(self, h_star, s, x):
        """

        Args:
            h_star: combined context vector over lemma and tag
            s: decoder hidden state, of shape (1, bsz, hidden_size)
            x: decoder input, of shape (bsz, embedding_size)

        Returns:
            p_gen: generation probabilty, of shape (bsz, )
        """
        bsz = h_star.shape[0]
        p_gen = self.W_h_star(h_star) + self.W_s(s.squeeze(0)) + self.W_x(x
            ) + self.b_attn.expand(bsz, -1)
        p_gen = F.sigmoid(p_gen.squeeze(1))
        return p_gen

    def init_parameters(self):
        stdv = 1 / math.sqrt(100)
        self.b_attn.data.uniform_(-stdv, stdv)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'embedding_size': 4, 'hidden_size': 4, 'h_star_size': 4}]
