import torch


class TorchClampOptionMaxMin(torch.nn.Module):

    def forward(self, x):
        return torch.clamp(x, min=-0.1, max=0.1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
