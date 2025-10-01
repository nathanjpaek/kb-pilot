import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def where(condition, if_true, if_false):
    """
    Torch equivalent of numpy.where.

    Parameters
    ----------
    condition : torch.ByteTensor or torch.cuda.ByteTensor or torch.autograd.Variable
        Condition to check.
    if_true : torch.Tensor or torch.cuda.Tensor or torch.autograd.Variable
        Output value if condition is true.
    if_false: torch.Tensor or torch.cuda.Tensor or torch.autograd.Variable
        Output value if condition is false

    Returns
    -------
    torch.Tensor

    Raises
    ------
    AssertionError
        if if_true and if_false are not both variables or both tensors.
    AssertionError
        if if_true and if_false don't have the same datatype.
    """
    if isinstance(if_true, Variable) or isinstance(if_false, Variable):
        assert isinstance(condition, Variable
            ), 'Condition must be a variable if either if_true or if_false is a variable.'
        assert isinstance(if_false, Variable) and isinstance(if_false, Variable
            ), 'Both if_true and if_false must be variables if either is one.'
        assert if_true.data.type() == if_false.data.type(
            ), 'Type mismatch: {} and {}'.format(if_true.data.type(),
            if_false.data.type())
    else:
        assert not isinstance(condition, Variable
            ), 'Condition must not be a variable because neither if_true nor if_false is one.'
        assert if_true.type() == if_false.type(
            ), 'Type mismatch: {} and {}'.format(if_true.data.type(),
            if_false.data.type())
    casted_condition = condition.type_as(if_true)
    output = casted_condition * if_true + (1 - casted_condition) * if_false
    return output


class SELU(nn.Module):

    def forward(self, input):
        return self.selu(input)

    @staticmethod
    def selu(x):
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        return scale * where(x >= 0, x, alpha * F.elu(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
