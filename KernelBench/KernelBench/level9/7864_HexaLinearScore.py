import math
import torch
import torch.utils.data.dataloader
import torch.nn as nn
import torch.nn


class HexaLinearScore(nn.Module):
    """
    Outer product version of hexalinear function for sequence labeling.
    """

    def __init__(self, wemb_size, tagset_size, temb_size=20, rank=396, std=
        0.1545, normalization=True, **kwargs):
        """
        Args:
            wemb_size: word embedding hidden size
            tagset_size: tag set size
            temb_size: tag embedding size
            rank: rank of the weight tensor
            std: standard deviation of the tensor
        """
        super(HexaLinearScore, self).__init__()
        self.wemb_size = wemb_size
        self.tagset_size = tagset_size
        self.temb_size = temb_size
        self.rank = rank
        self.std = std
        self.normalization = normalization
        self.tag_emd = nn.Parameter(torch.Tensor(self.tagset_size, self.
            temb_size))
        self.W1 = nn.Parameter(torch.Tensor(self.wemb_size, self.rank))
        self.W2 = nn.Parameter(torch.Tensor(self.wemb_size, self.rank))
        self.W3 = nn.Parameter(torch.Tensor(self.wemb_size, self.rank))
        self.T1 = nn.Parameter(torch.Tensor(self.temb_size, self.rank))
        self.T2 = nn.Parameter(torch.Tensor(self.temb_size, self.rank))
        self.T3 = nn.Parameter(torch.Tensor(self.temb_size, self.rank))
        self.rand_init()
        self

    def rand_init(self):
        """random initialization
        """
        nn.init.uniform_(self.tag_emd, a=math.sqrt(6 / self.temb_size), b=
            math.sqrt(6 / self.temb_size))
        nn.init.normal_(self.T1, std=self.std)
        nn.init.normal_(self.T2, std=self.std)
        nn.init.normal_(self.T3, std=self.std)
        nn.init.normal_(self.W1, std=self.std)
        nn.init.normal_(self.W2, std=self.std)
        nn.init.normal_(self.W3, std=self.std)

    def forward(self, word_emb):
        """
        Args:
            word_emb: [batch, sent_length, wemb_size]
        Returns: Tensor
            [batch, sent_length-window_size, tagset_size, tagset_size]
        """
        assert word_emb.size(2
            ) == self.wemb_size, 'batch sizes of encoder and decoder are requires to be equal.'
        g1 = torch.matmul(word_emb[:, :-2], self.W1)
        g2 = torch.matmul(word_emb[:, 1:-1], self.W2)
        g3 = torch.matmul(word_emb[:, 2:], self.W3)
        g4 = torch.matmul(self.tag_emd, self.T1)
        g5 = torch.matmul(self.tag_emd, self.T2)
        g6 = torch.matmul(self.tag_emd, self.T3)
        temp01 = g1 * g2 * g3
        temp02 = torch.einsum('ak,bk,ck->abck', [g4, g5, g6])
        score = torch.einsum('nmk,abck->nmabc', [temp01, temp02])
        if self.normalization:
            score = score / math.sqrt(self.rank)
        return score


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'wemb_size': 4, 'tagset_size': 4}]
