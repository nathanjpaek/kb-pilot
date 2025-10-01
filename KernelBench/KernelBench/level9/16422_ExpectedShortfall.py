from torch.nn import Module
import torch
from torch import Tensor
from typing import Callable
from typing import Union
from typing import Optional
from abc import ABC
from math import ceil


def bisect(fn: 'Callable[[Tensor], Tensor]', target: 'Tensor', lower:
    'Union[float, Tensor]', upper: 'Union[float, Tensor]', precision:
    'float'=1e-06, max_iter: 'int'=100000) ->Tensor:
    """Perform binary search over a tensor.

    The output tensor approximately satisfies the following relation:

    .. code-block::

        fn(output) = target

    Args:
        fn (callable[[Tensor], Tensor]): A monotone function.
        target (Tensor): Target of function values.
        lower (Tensor or float): Lower bound of binary search.
        upper (Tensor or float): Upper bound of binary search.
        precision (float, default=1e-6): Precision of output.
        max_iter (int, default 100000): If the number of iterations exceeds this
            value, abort computation and raise RuntimeError.

    Returns:
        torch.Tensor

    Raises:
        RuntimeError: If the number of iteration exceeds ``max_iter``.

    Examples:

        >>> target = torch.tensor([-1.0, 0.0, 1.0])
        >>> fn = torch.log
        >>> output = bisect(fn, target, 0.01, 10.0)
        >>> output
        tensor([0.3679, 1.0000, 2.7183])
        >>> torch.allclose(fn(output), target, atol=1e-6)
        True

        Monotone decreasing function:

        >>> fn = lambda input: -torch.log(input)
        >>> output = bisect(fn, target, 0.01, 10.0)
        >>> output
        tensor([2.7183, 1.0000, 0.3679])
        >>> torch.allclose(fn(output), target, atol=1e-6)
        True
    """
    lower, upper = map(torch.as_tensor, (lower, upper))
    if not (lower < upper).all():
        raise ValueError('condition lower < upper should be satisfied.')
    if (fn(lower) > fn(upper)).all():

        def mf(input):
            return -fn(input)
        return bisect(mf, -target, lower, upper, precision=precision,
            max_iter=max_iter)
    n_iter = 0
    while torch.max(upper - lower) > precision:
        n_iter += 1
        if n_iter > max_iter:
            raise RuntimeError(
                f'Aborting since iteration exceeds max_iter={max_iter}.')
        m = (lower + upper) / 2
        output = fn(m)
        lower = lower.where(output >= target, m)
        upper = upper.where(output < target, m)
    return upper


def topp(input: 'Tensor', p: 'float', dim: 'Optional[int]'=None, largest:
    'bool'=True):
    """Returns the largest :math:`p * N` elements of the given input tensor,
    where :math:`N` stands for the total number of elements in the input tensor.

    If ``dim`` is not given, the last dimension of the ``input`` is chosen.

    If ``largest`` is ``False`` then the smallest elements are returned.

    A namedtuple of ``(values, indices)`` is returned, where the ``indices``
    are the indices of the elements in the original ``input`` tensor.

    Args:
        input (torch.Tensor): The input tensor.
        p (float): The quantile level.
        dim (int, optional): The dimension to sort along.
        largest (bool, default=True): Controls whether to return largest or smallest
            elements.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import topp
        >>>
        >>> input = torch.arange(1.0, 6.0)
        >>> input
        tensor([1., 2., 3., 4., 5.])
        >>> topp(input, 3 / 5)
        torch.return_types.topk(
        values=tensor([5., 4., 3.]),
        indices=tensor([4, 3, 2]))
    """
    if dim is None:
        return input.topk(ceil(p * input.numel()), largest=largest)
    else:
        return input.topk(ceil(p * input.size(dim)), dim=dim, largest=largest)


def expected_shortfall(input: 'Tensor', p: 'float', dim: 'Optional[int]'=None
    ) ->Tensor:
    """Returns the expected shortfall of the given input tensor.

    Args:
        input (torch.Tensor): The input tensor.
        p (float): The quantile level.
        dim (int, optional): The dimension to sort along.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import expected_shortfall
        >>>
        >>> input = -torch.arange(10.0)
        >>> input
        tensor([-0., -1., -2., -3., -4., -5., -6., -7., -8., -9.])
        >>> expected_shortfall(input, 0.3)
        tensor(8.)
    """
    if dim is None:
        return -topp(input, p=p, largest=False).values.mean()
    else:
        return -topp(input, p=p, largest=False, dim=dim).values.mean(dim=dim)


class HedgeLoss(Module, ABC):
    """Base class for hedging criteria."""

    def forward(self, input: 'Tensor') ->Tensor:
        """Returns the loss of the profit-loss distribution.

        This method should be overridden.

        Args:
            input (torch.Tensor): The distribution of the profit and loss.

        Shape:
            - Input: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - Output: :math:`(*)`

        Returns:
            torch.Tensor
        """

    def cash(self, input: 'Tensor') ->Tensor:
        """Returns the cash amount which is as preferable as
        the given profit-loss distribution in terms of the loss.

        The output ``cash`` is expected to satisfy the following relation:

        .. code::

            loss(torch.full_like(pnl, cash)) = loss(pnl)

        By default, the output is computed by binary search.
        If analytic form is known, it is recommended to override this method
        for faster computation.

        Args:
            input (torch.Tensor): The distribution of the profit and loss.

        Shape:
            - Input: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - Output: :math:`(*)`

        Returns:
            torch.Tensor
        """
        return bisect(self, self(input), input.min(), input.max())


class ExpectedShortfall(HedgeLoss):
    """Creates a criterion that measures the expected shortfall.

    .. seealso::
        - :func:`pfhedge.nn.functional.expected_shortfall`

    Args:
        p (float, default=0.1): Quantile level.
            This parameter should satisfy :math:`0 < p \\leq 1`.

    Shape:
        - Input: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - Output: :math:`(*)`

    Examples:
        >>> from pfhedge.nn import ExpectedShortfall
        >>>
        >>> loss = ExpectedShortfall(0.5)
        >>> input = -torch.arange(4.0)
        >>> loss(input)
        tensor(2.5000)
        >>> loss.cash(input)
        tensor(-2.5000)
    """

    def __init__(self, p: 'float'=0.1):
        if not 0 < p <= 1:
            raise ValueError('The quantile level should satisfy 0 < p <= 1.')
        super().__init__()
        self.p = p

    def extra_repr(self) ->str:
        return str(self.p)

    def forward(self, input: 'Tensor') ->Tensor:
        return expected_shortfall(input, p=self.p, dim=0)

    def cash(self, input: 'Tensor') ->Tensor:
        return -self(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
