import torch
import torch.nn as nn
import torch.utils.data
from torch.nn.init import xavier_uniform_


class TorchGloVeModel(nn.Module):

    def __init__(self, n_words, embed_dim):
        super().__init__()
        self.n_words = n_words
        self.embed_dim = embed_dim
        self.W = self._init_weights(self.n_words, self.embed_dim)
        self.C = self._init_weights(self.n_words, self.embed_dim)
        self.bw = self._init_weights(self.n_words, 1)
        self.bc = self._init_weights(self.n_words, 1)

    def _init_weights(self, m, n):
        return nn.Parameter(xavier_uniform_(torch.empty(m, n)))

    def forward(self, X_log, idx):
        """
        Parameters
        ----------
        X_log : torch.FloatTensor, shape `(batch_size, n_vocab)`.

        idx : torch.LongTensor, shape `(batch_size, )`
            Indices of the vocab items in the current batch.

        Returns
        -------
        torch.FloatTensor, shape `(n_vocab, n_vocab)`.

        """
        preds = self.W[idx].matmul(self.C.T) + self.bw[idx] + self.bc.T
        diffs = preds - X_log
        return diffs


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'n_words': 4, 'embed_dim': 4}]
