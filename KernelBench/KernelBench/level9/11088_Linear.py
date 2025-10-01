import torch


class Linear(torch.nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = torch.nn.Parameter(2 * (torch.rand(in_size, out_size) -
            0.5))
        self.bias = torch.nn.Parameter(2 * (torch.rand(out_size) - 0.5))

    def forward(self, x):
        return x @ self.weight + self.bias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'out_size': 4}]
