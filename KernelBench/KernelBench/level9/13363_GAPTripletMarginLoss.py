import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
from torch.functional import F


def global_average_pooling(inp: 'torch.Tensor') ->torch.Tensor:
    if inp.ndim == 5:
        return F.adaptive_avg_pool3d(inp, 1)
    elif inp.ndim == 4:
        return F.adaptive_avg_pool2d(inp, 1)
    else:
        raise NotImplementedError


class GAPTripletMarginLoss(nn.TripletMarginLoss):
    """Same as ``torch.nn.TripletMarginLoss``, but applies global average
    pooling to anchor, positive and negative tensors before calculating the
    loss."""

    def forward(self, anchor: 'torch.Tensor', positive: 'torch.Tensor',
        negative: 'torch.Tensor') ->torch.Tensor:
        return super().forward(global_average_pooling(anchor),
            global_average_pooling(positive), global_average_pooling(negative))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
