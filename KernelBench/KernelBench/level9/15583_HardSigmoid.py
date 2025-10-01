import torch


def hard_sigmoid(tensor: 'torch.Tensor', inplace: 'bool'=False) ->torch.Tensor:
    """
    Applies HardSigmoid function element-wise.

    See :class:`torchlayers.activations.HardSigmoid` for more details.

    Arguments:
        tensor :
            Tensor activated element-wise
        inplace :
            Whether operation should be performed `in-place`. Default: `False`

    Returns:
        torch.Tensor:
    """
    return torch.nn.functional.hardtanh(tensor, min_val=0, inplace=inplace)


class HardSigmoid(torch.nn.Module):
    """
    Applies HardSigmoid function element-wise.

    Uses `torch.nn.functional.hardtanh` internally with `0` and `1` ranges.

    Arguments:
        tensor :
            Tensor activated element-wise

    """

    def forward(self, tensor: 'torch.Tensor'):
        return hard_sigmoid(tensor)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
