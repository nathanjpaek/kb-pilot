import torch
import torch.nn as nn


class generator(nn.Module):

    def __init__(self, p_dim, c_dim):
        super(generator, self).__init__()
        self.p_dim = p_dim
        self.c_dim = c_dim
        convex_layer_weights = torch.zeros((self.p_dim, self.c_dim))
        self.convex_layer_weights = nn.Parameter(convex_layer_weights)
        nn.init.normal_(self.convex_layer_weights, mean=0.0, std=0.02)

    def forward(self, points, plane_m, is_training=False):
        h1 = torch.matmul(points, plane_m)
        h1 = torch.clamp(h1, min=0)
        h2 = torch.matmul(h1, (self.convex_layer_weights > 0.01).float())
        h3 = torch.min(h2, dim=2, keepdim=True)[0]
        return h2, h3


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'p_dim': 4, 'c_dim': 4}]
