import torch
import torch.nn as nn


class Probe(nn.Module):
    pass


class TwoWordPSDProbe(Probe):
    """ Computes squared L2 distance after projection by a matrix.
    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """

    def __init__(self, model_dim, probe_rank=1024):
        None
        super(TwoWordPSDProbe, self).__init__()
        self.probe_rank = probe_rank
        self.model_dim = model_dim
        self.proj = nn.Parameter(data=torch.zeros(self.model_dim, self.
            probe_rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)

    def forward(self, batch):
        """ Computes all n^2 pairs of distances after projection
        for each sentence in a batch.
        Note that due to padding, some distances will be non-zero for pads.
        Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j
        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)
        _batchlen, seqlen, _rank = transformed.size()
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1, 2)
        diffs = transformed - transposed
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)
        return squared_distances


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'model_dim': 4}]
