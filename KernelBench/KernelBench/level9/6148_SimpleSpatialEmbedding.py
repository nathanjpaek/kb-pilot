import torch
import torch.nn


class SimpleSpatialEmbedding(torch.nn.Module):

    def __init__(self, in_features, out_features, weight_multiplier=1.0):
        super(SimpleSpatialEmbedding, self).__init__()
        self.b = torch.zeros((in_features, out_features))
        self.b.normal_(0, weight_multiplier)
        self.b = torch.nn.Parameter(2.0 ** self.b - 1)
        self.osize = out_features

    def forward(self, x):
        x = torch.matmul(x, self.b)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

    def output_size(self):
        return 2 * self.osize


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
