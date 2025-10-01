import torch
from torch import nn


class LabelSmoothing(nn.Module):
    """
    Label Smoothing

    Attributes
    ----------
    criterion   : torch.nn.KLDivLoss
    padding_idx : int
    eps         : float
    n_vocab     : int
    """

    def __init__(self, n_vocab, eps, padding_idx=0):
        """
        Parameters
        ----------
        n_vocab     : int
            size of vocab
        eps         : float
            portion of the one hot that will be taken away
        padding_idx : int
            indicator of which value should be considered as padding
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.eps = eps
        self.n_vocab = n_vocab

    def forward(self, pred, gold):
        """
        Parameters
        ----------
        pred : 2d tensor (batch_size * seq_len, n_vocab)
        gold : 1d tensor of int (batch_size * seq_len)
        """
        n_vocab, eps, padding_idx = self.n_vocab, self.eps, self.padding_idx
        dist = pred.clone()
        dist.fill_(eps / (n_vocab - 2))
        dist.scatter_(1, gold.view(gold.shape[0], -1), 1 - eps)
        dist[:, padding_idx] = 0
        mask = gold != padding_idx
        dist.mul_(mask.view(dist.shape[0], 1))
        return self.criterion(pred, dist)


def get_inputs():
    return [torch.rand([4, 2]), torch.ones([4, 1], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'n_vocab': 4, 'eps': 4}]
