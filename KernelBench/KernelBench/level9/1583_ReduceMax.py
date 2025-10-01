import torch


class ReduceMax(torch.nn.Module):

    def forward(self, inputs, mask=None):
        return torch.amax(inputs, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
