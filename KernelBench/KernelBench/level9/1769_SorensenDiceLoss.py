import torch
import torch.nn as nn
from torch.autograd import Variable


def assert_(condition, message='', exception_type=AssertionError):
    """Like assert, but with arbitrary exception types."""
    if not condition:
        raise exception_type(message)


def flatten_samples(tensor_or_variable):
    """
    Flattens a tensor or a variable such that the channel axis is first and the sample axis
    is second. The shapes are transformed as follows:
        (N, C, H, W) --> (C, N * H * W)
        (N, C, D, H, W) --> (C, N * D * H * W)
        (N, C) --> (C, N)
    The input must be atleast 2d.
    """
    assert_(tensor_or_variable.dim() >= 2,
        'Tensor or variable must be atleast 2D. Got one of dim {}.'.format(
        tensor_or_variable.dim()), ShapeError)
    num_channels = tensor_or_variable.size(1)
    permute_axes = list(range(tensor_or_variable.dim()))
    permute_axes[0], permute_axes[1] = permute_axes[1], permute_axes[0]
    permuted = tensor_or_variable.permute(*permute_axes).contiguous()
    flattened = permuted.view(num_channels, -1)
    return flattened


class ShapeError(ValueError):
    pass


class SorensenDiceLoss(nn.Module):
    """
    Computes a loss scalar, which when minimized maximizes the Sorensen-Dice similarity
    between the input and the target. For both inputs and targets it must be the case that
    `input_or_target.size(1) = num_channels`.
    """

    def __init__(self, weight=None, channelwise=True, eps=1e-06):
        """
        Parameters
        ----------
        weight : torch.FloatTensor or torch.cuda.FloatTensor
            Class weights. Applies only if `channelwise = True`.
        channelwise : bool
            Whether to apply the loss channelwise and sum the results (True)
            or to apply it on all channels jointly (False).
        """
        super(SorensenDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.channelwise = channelwise
        self.eps = eps

    def forward(self, input, target):
        if not self.channelwise:
            numerator = (input * target).sum()
            denominator = (input * input).sum() + (target * target).sum()
            loss = -2.0 * (numerator / denominator.clamp(min=self.eps))
        else:
            input = flatten_samples(input)
            target = flatten_samples(target)
            numerator = (input * target).sum(-1)
            denominator = (input * input).sum(-1) + (target * target).sum(-1)
            channelwise_loss = -2 * (numerator / denominator.clamp(min=self
                .eps))
            if self.weight is not None:
                if channelwise_loss.dim() == 2:
                    channelwise_loss = channelwise_loss.squeeze(1)
                weight = Variable(self.weight, requires_grad=False)
                channelwise_loss = weight * channelwise_loss
            loss = channelwise_loss.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
