import torch


class TensorProba(torch.nn.Module):

    def __init__(self, dim=1):
        self.dim = dim
        super().__init__()

    def forward(self, input):
        total = torch.sum(input, dim=self.dim, keepdim=True)
        return input / total


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
