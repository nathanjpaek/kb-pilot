import torch
import torch.nn as nn
import torch.utils.checkpoint


class PositionalEncoding2D(nn.Module):

    def __init__(self, d_model, minpos=-32, maxpos=32, p_drop=0.1):
        super(PositionalEncoding2D, self).__init__()
        self.minpos = minpos
        self.maxpos = maxpos
        self.nbin = abs(minpos) + maxpos + 1
        self.emb = nn.Embedding(self.nbin, d_model)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x, idx):
        bins = torch.arange(self.minpos, self.maxpos, device=x.device)
        seqsep = idx[:, None, :] - idx[:, :, None]
        ib = torch.bucketize(seqsep, bins).long()
        emb = self.emb(ib)
        x = x + emb
        return self.drop(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
