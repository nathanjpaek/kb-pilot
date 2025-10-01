import torch


class TorchClampOptionMax(torch.nn.Module):

    def forward(self, x):
        return torch.clamp(x, max=0.1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
