from torch.nn import Module
import torch
from torch import Tensor
from typing import Callable
from typing import Union
from abc import ABC


def _format_float(value: 'float') ->str:
    """
    >>> _format_float(1)
    '1'
    >>> _format_float(1.0)
    '1.'
    >>> _format_float(1e-4)
    '1.0000e-04'
    """
    tensor = torch.tensor([value])
    return torch._tensor_str._Formatter(tensor).format(value)


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


def exp_utility(input: 'Tensor', a: 'float'=1.0) ->Tensor:
    """Applies an exponential utility function.

    An exponential utility function is defined as:

    .. math::

        u(x) = -\\exp(-a x) \\,.

    Args:
        input (torch.Tensor): The input tensor.
        a (float, default=1.0): The risk aversion coefficient of the exponential
            utility.

    Returns:
        torch.Tensor
    """
    return -(-a * input).exp()


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


class EntropicLoss(HedgeLoss):
    """Creates a criterion that measures the expected exponential utility.

    The loss of the profit-loss :math:`\\text{pnl}` is given by:

    .. math::
        \\text{loss}(\\text{pnl}) = -\\mathbf{E}[u(\\text{pnl})] \\,,
        \\quad
        u(x) = -\\exp(-a x) \\,.

    .. seealso::
        - :func:`pfhedge.nn.functional.exp_utility`:
          The corresponding utility function.

    Args:
        a (float > 0, default=1.0): Risk aversion coefficient of
            the exponential utility.

    Shape:
        - Input: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - Output: :math:`(*)`

    Examples:
        >>> from pfhedge.nn import EntropicLoss
        >>>
        >>> loss = EntropicLoss()
        >>> input = -torch.arange(4.0)
        >>> loss(input)
        tensor(7.7982)
        >>> loss.cash(input)
        tensor(-2.0539)
    """

    def __init__(self, a: 'float'=1.0):
        if not a > 0:
            raise ValueError('Risk aversion coefficient should be positive.')
        super().__init__()
        self.a = a

    def extra_repr(self) ->str:
        return 'a=' + _format_float(self.a) if self.a != 1 else ''

    def forward(self, input: 'Tensor') ->Tensor:
        return -exp_utility(input, a=self.a).mean(0)

    def cash(self, input: 'Tensor') ->Tensor:
        return -(-exp_utility(input, a=self.a).mean(0)).log() / self.a


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
