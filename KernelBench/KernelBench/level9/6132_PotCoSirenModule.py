import torch
import torch.nn


class PotCoSirenModule(torch.nn.Module):

    def __init__(self, in_features, out_features, weight_multiplier=1.0):
        super(PotCoSirenModule, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features // 2)
        torch.nn.init.uniform_(self.linear.weight, a=-weight_multiplier, b=
            weight_multiplier)
        self.linear.weight.data = 2 ** self.linear.weight.data

    def forward(self, x):
        x = self.linear(x)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
