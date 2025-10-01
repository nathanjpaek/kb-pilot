import torch
import torch.nn as nn


class TemporalFusion(nn.Module):

    def __init__(self, nf, n_frame):
        super(TemporalFusion, self).__init__()
        self.n_frame = n_frame
        self.ref_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.nbr_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.up_conv = nn.Conv2d(nf * n_frame, nf * 4, 1, 1, bias=True)
        self.ps = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()
        emb_ref = self.ref_conv(x[:, N // 2, :, :, :].clone())
        emb = self.nbr_conv(x.view(-1, C, H, W)).view(B, N, C, H, W)
        cor_l = []
        for i in range(N):
            cor = torch.sum(emb[:, i, :, :, :] * emb_ref, dim=1, keepdim=True)
            cor_l.append(cor)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W
            )
        aggr_fea = x.view(B, -1, H, W) * cor_prob
        fea = self.lrelu(self.up_conv(aggr_fea))
        out = self.ps(fea)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nf': 4, 'n_frame': 4}]
