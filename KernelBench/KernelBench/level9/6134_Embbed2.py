import torch
import torch.nn


class Embbed2(torch.nn.Module):

    def __init__(self, in_features, out_features, weight_multiplier=1.0):
        super(Embbed2, self).__init__()
        self.b = 2.0 ** torch.linspace(0, weight_multiplier, out_features //
            in_features) - 1
        self.b = torch.nn.Parameter(torch.reshape(torch.eye(in_features) *
            self.b[:, None, None], [out_features, in_features]))
        self.osize = out_features
        self.a = torch.nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        x = torch.matmul(x, self.b.T)
        return torch.cat([self.a * torch.sin(x), self.a * torch.cos(x)], dim=-1
            )

    def output_size(self):
        return 2 * self.osize


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
