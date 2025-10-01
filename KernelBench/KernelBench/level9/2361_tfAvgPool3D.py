import torch
from torch import Tensor
from torch import nn


class tfAvgPool3D(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.avgf = nn.AvgPool3d((1, 3, 3), stride=(1, 2, 2))

    def forward(self, x: 'Tensor') ->Tensor:
        if x.shape[-1] != x.shape[-2]:
            raise RuntimeError('only same shape for h and w ' +
                'are supported by avg with tf_like')
        if x.shape[-1] != x.shape[-2]:
            raise RuntimeError('only same shape for h and w ' +
                'are supported by avg with tf_like')
        f1 = x.shape[-1] % 2 != 0
        if f1:
            padding_pad = 0, 0, 0, 0
        else:
            padding_pad = 0, 1, 0, 1
        x = torch.nn.functional.pad(x, padding_pad)
        if f1:
            x = torch.nn.functional.avg_pool3d(x, (1, 3, 3), stride=(1, 2, 
                2), count_include_pad=False, padding=(0, 1, 1))
        else:
            x = self.avgf(x)
            x[..., -1] = x[..., -1] * 9 / 6
            x[..., -1, :] = x[..., -1, :] * 9 / 6
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
