import torch


class MaxRotationPoolP4(torch.nn.Module):

    def forward(self, x):
        return x.max(2).values


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
