import torch
import torch.nn as nn
import torch.fft


class MiniBatchAverageLayer(nn.Module):
    """Minibatch stat concatenation layer. Implementation is from https://github.com/shanexn/pytorch-pggan."""

    def __init__(self, offset=1e-08):
        super().__init__()
        self.offset = offset

    def forward(self, x):
        stddev = torch.sqrt(torch.mean((x - torch.mean(x, dim=0, keepdim=
            True)) ** 2, dim=0, keepdim=True) + self.offset)
        inject_shape = list(x.size())[:]
        inject_shape[1] = 1
        inject = torch.mean(stddev, dim=1, keepdim=True)
        inject = inject.expand(inject_shape)
        return torch.cat((x, inject), dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
