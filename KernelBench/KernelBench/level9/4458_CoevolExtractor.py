import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-05):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps
            )
        x = self.a_2 * (x - mean)
        x /= std
        x += self.b_2
        return x


class CoevolExtractor(nn.Module):

    def __init__(self, n_feat_proj, n_feat_out, p_drop=0.1):
        super(CoevolExtractor, self).__init__()
        self.norm_2d = LayerNorm(n_feat_proj * n_feat_proj)
        self.proj_2 = nn.Linear(n_feat_proj ** 2, n_feat_out)

    def forward(self, x_down, x_down_w):
        B, _N, L = x_down.shape[:3]
        pair = torch.einsum('abij,ablm->ailjm', x_down, x_down_w)
        pair = pair.reshape(B, L, L, -1)
        pair = self.norm_2d(pair)
        pair = self.proj_2(pair)
        return pair


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_feat_proj': 4, 'n_feat_out': 4}]
