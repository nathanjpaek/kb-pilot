import torch
from typing import Tuple
from torch.nn.modules.loss import _Loss
from typing import List
from typing import Optional


def _reduce(x: 'torch.Tensor', reduction: 'str'='mean') ->torch.Tensor:
    """Reduce input in batch dimension if needed.

    Args:
        x: Tensor with shape (N, *).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
    """
    if reduction == 'none':
        return x
    elif reduction == 'mean':
        return x.mean(dim=0)
    elif reduction == 'sum':
        return x.sum(dim=0)
    else:
        raise ValueError(
            "Uknown reduction. Expected one of {'none', 'mean', 'sum'}")


def _validate_input(tensors: 'List[torch.Tensor]', dim_range:
    'Tuple[int, int]'=(0, -1), data_range: 'Tuple[float, float]'=(0.0, -1.0
    ), size_range: 'Optional[Tuple[int, int]]'=None) ->None:
    """Check that input(-s)  satisfies the requirements
    Args:
        tensors: Tensors to check
        dim_range: Allowed number of dimensions. (min, max)
        data_range: Allowed range of values in tensors. (min, max)
        size_range: Dimensions to include in size comparison. (start_dim, end_dim + 1)
    """
    if not __debug__:
        return
    x = tensors[0]
    for t in tensors:
        assert torch.is_tensor(t), f'Expected torch.Tensor, got {type(t)}'
        assert t.device == x.device, f'Expected tensors to be on {x.device}, got {t.device}'
        if size_range is None:
            assert t.size() == x.size(
                ), f'Expected tensors with same size, got {t.size()} and {x.size()}'
        else:
            assert t.size()[size_range[0]:size_range[1]] == x.size()[size_range
                [0]:size_range[1]
                ], f'Expected tensors with same size at given dimensions, got {t.size()} and {x.size()}'
        if dim_range[0] == dim_range[1]:
            assert t.dim() == dim_range[0
                ], f'Expected number of dimensions to be {dim_range[0]}, got {t.dim()}'
        elif dim_range[0] < dim_range[1]:
            assert dim_range[0] <= t.dim() <= dim_range[1
                ], f'Expected number of dimensions to be between {dim_range[0]} and {dim_range[1]}, got {t.dim()}'
        if data_range[0] < data_range[1]:
            assert data_range[0] <= t.min(
                ), f'Expected values to be greater or equal to {data_range[0]}, got {t.min()}'
            assert t.max() <= data_range[1
                ], f'Expected values to be lower or equal to {data_range[1]}, got {t.max()}'


def total_variation(x: 'torch.Tensor', reduction: 'str'='mean', norm_type:
    'str'='l2') ->torch.Tensor:
    """Compute Total Variation metric

    Args:
        x: Tensor. Shape :math:`(N, C, H, W)`.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        norm_type: ``'l1'`` | ``'l2'`` | ``'l2_squared'``,
            defines which type of norm to implement, isotropic  or anisotropic.

    Returns:
        Total variation of a given tensor

    References:
        https://www.wikiwand.com/en/Total_variation_denoising

        https://remi.flamary.com/demos/proxtv.html
    """
    _validate_input([x], dim_range=(4, 4), data_range=(0, -1))
    if norm_type == 'l1':
        w_variance = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]),
            dim=[1, 2, 3])
        h_variance = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]),
            dim=[1, 2, 3])
        score = h_variance + w_variance
    elif norm_type == 'l2':
        w_variance = torch.sum(torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 
            2), dim=[1, 2, 3])
        h_variance = torch.sum(torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 
            2), dim=[1, 2, 3])
        score = torch.sqrt(h_variance + w_variance)
    elif norm_type == 'l2_squared':
        w_variance = torch.sum(torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 
            2), dim=[1, 2, 3])
        h_variance = torch.sum(torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 
            2), dim=[1, 2, 3])
        score = h_variance + w_variance
    else:
        raise ValueError(
            "Incorrect norm type, should be one of {'l1', 'l2', 'l2_squared'}")
    return _reduce(score, reduction)


class TVLoss(_Loss):
    """Creates a criterion that measures the total variation of the
    the given input :math:`x`.


    If :attr:`norm_type` set to ``'l2'`` the loss can be described as:

    .. math::
        TV(x) = \\sum_{N}\\sqrt{\\sum_{H, W, C}(|x_{:, :, i+1, j} - x_{:, :, i, j}|^2 +
        |x_{:, :, i, j+1} - x_{:, :, i, j}|^2)}

    Else if :attr:`norm_type` set to ``'l1'``:

    .. math::
        TV(x) = \\sum_{N}\\sum_{H, W, C}(|x_{:, :, i+1, j} - x_{:, :, i, j}| +
        |x_{:, :, i, j+1} - x_{:, :, i, j}|)

    where :math:`N` is the batch size, `C` is the channel size.

    Args:
        norm_type: one of ``'l1'`` | ``'l2'`` | ``'l2_squared'``
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``

    Examples:

        >>> loss = TVLoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> output = loss(x)
        >>> output.backward()

    References:
        https://www.wikiwand.com/en/Total_variation_denoising

        https://remi.flamary.com/demos/proxtv.html
    """

    def __init__(self, norm_type: 'str'='l2', reduction: 'str'='mean'):
        super().__init__()
        self.norm_type = norm_type
        self.reduction = reduction

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Computation of Total Variation (TV) index as a loss function.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of TV loss to be minimized.
        """
        score = total_variation(x, reduction=self.reduction, norm_type=self
            .norm_type)
        return score


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
