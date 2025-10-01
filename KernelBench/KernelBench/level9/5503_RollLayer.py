import torch


class RollLayer(torch.nn.Module):
    """
        Layer which shifts the dimensions for performing the coupling permutations
        on different dimensions
    """

    def __init__(self, shift):
        super(RollLayer, self).__init__()
        self.shift = shift

    def forward(self, x):
        return torch.cat((torch.roll(x[:, :-1], self.shift, dims=-1), x[:, 
            -1:]), axis=-1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'shift': 4}]
