import math
import torch
import torch.nn as nn
import torch.utils.data.dataloader
import torch.nn


class QuadriLinearScore(nn.Module):
    """
    Outer product version of quadrilinear function for sequence labeling.
    """

    def __init__(self, wemb_size, tagset_size, temb_size=20, rank=396, std=
        0.1545, window_size=1, normalization=True, **kwargs):
        """
        Args:
            wemb_size: word embedding hidden size
            tagset_size: tag set size
            temb_size: tag embedding size
            rank: rank of the weight tensor
            std: standard deviation of the tensor
        """
        super(QuadriLinearScore, self).__init__()
        self.wemb_size = wemb_size
        self.tagset_size = tagset_size
        self.temb_size = temb_size
        self.rank = rank
        self.std = std
        self.window_size = window_size
        self.normalization = normalization
        self.tag_emd = nn.Parameter(torch.Tensor(self.tagset_size, self.
            temb_size))
        self.T = nn.Parameter(torch.Tensor(self.wemb_size, self.rank))
        self.U = nn.Parameter(torch.Tensor(self.wemb_size, self.rank))
        self.V = nn.Parameter(torch.Tensor(self.temb_size, self.rank))
        self.W = nn.Parameter(torch.Tensor(self.temb_size, self.rank))
        self.rand_init()
        self

    def rand_init(self):
        """random initialization
        """
        nn.init.uniform_(self.tag_emd, a=math.sqrt(6 / self.temb_size), b=
            math.sqrt(6 / self.temb_size))
        nn.init.normal_(self.T, std=self.std)
        nn.init.normal_(self.U, std=self.std)
        nn.init.normal_(self.V, std=self.std)
        nn.init.normal_(self.W, std=self.std)

    def forward(self, word_emb):
        """
        Args:
            word_emb: [batch, sent_length, wemb_size]
        Returns: Tensor
            [batch, sent_length-window_size, tagset_size, tagset_size]
        """
        assert word_emb.size(2
            ) == self.wemb_size, 'batch sizes of encoder and decoder are requires to be equal.'
        g0 = torch.matmul(word_emb[:, :-self.window_size], self.U)
        g1 = torch.matmul(word_emb[:, self.window_size:], self.T)
        g2 = torch.matmul(self.tag_emd, self.V)
        g3 = torch.matmul(self.tag_emd, self.W)
        temp01 = g0 * g1
        temp012 = torch.einsum('nak,bk->nabk', [temp01, g2])
        score = torch.einsum('nabk,ck->nabc', [temp012, g3])
        if self.normalization:
            score = score / math.sqrt(self.rank)
        return score


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'wemb_size': 4, 'tagset_size': 4}]
