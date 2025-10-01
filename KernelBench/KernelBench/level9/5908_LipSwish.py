import torch


class LipSwish(torch.nn.Module):

    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
