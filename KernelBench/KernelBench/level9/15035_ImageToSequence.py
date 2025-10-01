import torch
from typing import NamedTuple
from torch.nn.utils.rnn import pack_padded_sequence


def image_to_sequence(x, columnwise=True, return_packed=False):
    x, xs = (x.data, x.sizes) if isinstance(x, PaddedTensor) else (x, None)
    if x.dim() == 2:
        x = x.view(1, 1, x.size(0), x.size(1))
    elif x.dim() == 3:
        x = x.view(1, x.size(0), x.size(1), x.size(2))
    assert x.dim() == 4
    n, c, h, w = x.size()
    if columnwise:
        x = x.permute(3, 0, 1, 2).contiguous().view(w, n, h * c)
    else:
        x = x.permute(2, 0, 1, 3).contiguous().view(h, n, w * c)
    if xs is None:
        return x
    xs = xs[:, 1 if columnwise else 0]
    return pack_padded_sequence(x, xs.tolist()) if return_packed else (x,
        xs.tolist())


class PaddedTensor(NamedTuple):
    data: 'torch.Tensor'
    sizes: 'torch.Tensor'

    @classmethod
    def build(cls, data: 'torch.Tensor', sizes: 'torch.Tensor'):
        assert isinstance(data, torch.Tensor)
        assert isinstance(sizes, torch.Tensor)
        assert sizes.dim() == 2, 'PaddedTensor.sizes must have 2 dimensions'
        assert sizes.size(1) in (2, 3
            ), f'PaddedTensor.sizes is incorrect: expected=2 (HxW) or 3 (CxHxW), found={sizes.size(1)}'
        assert data.size(0) == sizes.size(0
            ), f'Batch size {sizes.size(0)} does not match the number of samples in the batch {data.size(0)}'
        return cls(data, sizes)

    def __repr__(self) ->str:
        return (
            f'PaddedTensor(data.size()={list(self.data.size())}, sizes={self.sizes.tolist()}, device={str(self.data.device)})'
            )

    @property
    def device(self) ->torch.device:
        return self.data.device


class ImageToSequence(torch.nn.Module):

    def __init__(self, columnwise=True, return_packed=False):
        super().__init__()
        self._columnwise = columnwise
        self._return_packed = return_packed

    def forward(self, x):
        return image_to_sequence(x, columnwise=self._columnwise,
            return_packed=self._return_packed)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
