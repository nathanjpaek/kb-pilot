import torch
import torch.nn as nn


class MinibatchStddev(nn.Module):
    """Minibatch Stddev layer from Progressive GAN"""

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        stddev_map = torch.sqrt(x.var(dim=0) + 1e-08).mean()
        stddev = stddev_map.expand(x.shape[0], 1, *x.shape[2:])
        return torch.cat([x, stddev], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
