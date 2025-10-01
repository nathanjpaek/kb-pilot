import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.tgt_gm = None

    def gram_matrix(self, x):
        a, b, c, d = x.shape
        features = x.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def forward(self, c_feats, s_feats, st_feats):
        c_loss = F.mse_loss(c_feats[2], st_feats[2], reduction='mean')
        s_loss = 0
        for ix in range(len(s_feats)):
            s_loss += F.mse_loss(self.gram_matrix(s_feats[ix]), self.
                gram_matrix(st_feats[ix]))
        return c_loss + 10000000.0 * s_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4, 4]), torch.
        rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
