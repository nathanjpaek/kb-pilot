import math
import torch
from torch import nn
from torch import autograd
from torch.nn import init


class ExtremeLinearFunction(autograd.Function):

    @staticmethod
    def forward(ctx, input, forward_weight, feedback_weight):
        ctx.save_for_backward(input, forward_weight, feedback_weight)
        output = input.mm(forward_weight.t())
        ctx.output = output
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, forward_weight, _feedback_weight = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[1]:
            inv_x = torch.pinverse(input)
            error_weight = inv_x.mm(ctx.output - grad_output)
            grad_weight = forward_weight - error_weight.t()
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(forward_weight)
        return grad_input, grad_weight, None


class ExtremeLinear(nn.Module):

    def __init__(self, input_features, output_features):
        super(ExtremeLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.forward_weight = nn.Parameter(torch.Tensor(output_features,
            input_features))
        self.feedback_weight = nn.Parameter(torch.Tensor(output_features,
            input_features))
        self.reset_parameters()

    def reset_parameters(self) ->None:
        init.kaiming_uniform_(self.forward_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.feedback_weight, a=math.sqrt(5))

    def forward(self, input):
        return ExtremeLinearFunction.apply(input, self.forward_weight, self
            .feedback_weight)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_features': 4, 'output_features': 4}]
