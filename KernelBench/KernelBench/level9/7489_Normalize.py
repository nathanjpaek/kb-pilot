import torch
import torch.utils.data


class Normalize(torch.nn.Module):

    def __init__(self):
        super(Normalize, self).__init__()
        self.normalize = torch.nn.functional.normalize

    def forward(self, x):
        x = self.normalize(x, dim=-1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
