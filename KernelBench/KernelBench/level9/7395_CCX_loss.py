import torch
import torch.utils.data
import torch.nn as nn


class CCX_loss(nn.Module):

    def __init__(self, eps=1e-06, h=0.5):
        super(CCX_loss, self).__init__()
        self.eps = eps
        self.h = h

    def forward(self, x, y):
        N, C, _H, _W = x.size()
        y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)
        x_centered = x - y_mu
        y_centered = y - y_mu
        x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1,
            keepdim=True)
        y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1,
            keepdim=True)
        x_normalized = x_normalized.reshape(N, C, -1)
        y_normalized = y_normalized.reshape(N, C, -1)
        cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)
        d = 1 - cosine_sim
        d_min, _ = torch.min(d, dim=2, keepdim=True)
        d_tilde = d / (d_min + self.eps)
        w = torch.exp((1 - d_tilde) / self.h)
        ccx_ij = w / torch.sum(w, dim=2, keepdim=True)
        ccx = torch.mean(torch.max(ccx_ij, dim=1)[0], dim=1)
        ccx_loss = torch.mean(-torch.log(ccx + self.eps))
        return ccx_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
