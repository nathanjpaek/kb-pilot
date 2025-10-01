import torch
import torch.nn as nn
import torch.utils.data.dataloader


class TwoWordBilinearLabelProbe(nn.Module):
    """ Computes a bilinear function of pairs of vectors.
    For a batch of sentences, computes all n^2 pairs of scores
    for each sentence in the batch.
    """

    def __init__(self, model_dim, rank, prob, device):
        super(TwoWordBilinearLabelProbe, self).__init__()
        self.maximum_rank = rank
        self.model_dim = model_dim
        self.proj_L = nn.Parameter(data=torch.zeros(self.model_dim, self.
            maximum_rank))
        self.proj_R = nn.Parameter(data=torch.zeros(self.maximum_rank, self
            .model_dim))
        self.bias = nn.Parameter(data=torch.zeros(1))
        nn.init.uniform_(self.proj_L, -0.05, 0.05)
        nn.init.uniform_(self.proj_R, -0.05, 0.05)
        nn.init.uniform_(self.bias, -0.05, 0.05)
        self
        self.dropout = nn.Dropout(p=prob)

    def forward(self, batch):
        """ Computes all n^2 pairs of attachment scores
        for each sentence in a batch.
        Computes h_i^TAh_j for all i,j
        where A = LR, L in R^{model_dim x maximum_rank}; R in R^{maximum_rank x model_rank}
        hence A is rank-constrained to maximum_rank.
        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of scores of shape (batch_size, max_seq_len, max_seq_len)
        """
        batchlen, seqlen, rank = batch.size()
        batch = self.dropout(batch)
        proj = torch.mm(self.proj_L, self.proj_R)
        batch_square = batch.unsqueeze(2).expand(batchlen, seqlen, seqlen, rank
            )
        batch_transposed = batch.unsqueeze(1).expand(batchlen, seqlen,
            seqlen, rank).contiguous().view(batchlen * seqlen * seqlen, rank, 1
            )
        psd_transformed = torch.matmul(batch_square.contiguous(), proj).view(
            batchlen * seqlen * seqlen, 1, rank)
        logits = (torch.bmm(psd_transformed, batch_transposed) + self.bias
            ).view(batchlen, seqlen, seqlen)
        return logits


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'model_dim': 4, 'rank': 4, 'prob': 0.5, 'device': 0}]
