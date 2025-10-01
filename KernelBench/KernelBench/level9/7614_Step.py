import torch
import torch.nn as nn


class StepF(torch.autograd.Function):
    """ A step function that returns values in {-1, 1} and uses the Straigh-Through Estimator
        to update upstream weights in the network
    """

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = torch.sign(input_).clamp(min=0) * 2 - 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        return grad_input


class Step(nn.Module):
    """Module wrapper for a step function (StepF).
    """

    def __init__(self):
        super(Step, self).__init__()

    def __repr__(self):
        s = '{name}(low=-1, high=1)'
        return s.format(name=self.__class__.__name__)

    def forward(self, x):
        return StepF.apply(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
