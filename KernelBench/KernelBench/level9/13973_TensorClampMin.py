import torch


class TensorClampMin(torch.nn.Module):

    def forward(self, x):
        return x.clamp_min(-0.1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
