import torch


class TensorClamp(torch.nn.Module):

    def forward(self, x):
        return x.clamp(-0.1, 0.1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
