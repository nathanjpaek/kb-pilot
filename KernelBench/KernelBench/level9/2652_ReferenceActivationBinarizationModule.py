import torch
from torch import nn
from torchvision import models as models
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.onnx


def get_per_channel_scale_shape(input_shape, is_weights):
    scale_shape = [(1) for _ in input_shape]
    if is_weights:
        scale_shape[0] = input_shape[0]
    else:
        scale_shape[1] = input_shape[1]
    elements = 1
    for i in scale_shape:
        elements *= i
    if elements == 1:
        return 1
    return scale_shape


def get_test_scale(num_channels):
    torch.manual_seed(0)
    retval = torch.Tensor(num_channels)
    retval.random_(0, 1)
    return retval


def get_test_threshold(input_shape):
    torch.manual_seed(0)
    threshold_shape = get_per_channel_scale_shape(input_shape, is_weights=False
        )
    retval = torch.Tensor(torch.zeros(threshold_shape))
    retval.random_(-10, 10)
    return retval


class ReferenceActivationBinarize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, scale, threshold):
        shape = [(1) for s in input_.shape]
        shape[1] = input_.shape[1]
        t = (threshold * scale).view(shape)
        output = (input_ > t).type(input_.dtype) * scale
        ctx.save_for_backward(input_, scale, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, scale, output = ctx.saved_variables
        mask_lower = (input_ <= scale).type(input_.dtype)
        grad_input = grad_output * (input_ >= 0).type(input_.dtype
            ) * mask_lower
        err = (output - input_) * scale.reciprocal()
        grad_scale = grad_output * (mask_lower * err + (1 - mask_lower))
        grad_scale = grad_scale.sum().view(1)
        grad_threshold = -grad_output * (input_ > 0).type(input_.dtype) * (
            input_ < scale).type(input_.dtype)
        for idx, _ in enumerate(input_.shape):
            if idx != 1:
                grad_threshold = grad_threshold.sum(idx, keepdim=True)
        return grad_input, grad_scale, grad_threshold


class ReferenceActivationBinarizationModule(nn.Module):

    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.scale = torch.nn.Parameter(get_test_scale(num_channels=1))
        self.threshold = torch.nn.Parameter(get_test_threshold(input_shape))

    def forward(self, input_):
        return ReferenceActivationBinarize.apply(input_, self.scale, self.
            threshold)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_shape': [4, 4]}]
