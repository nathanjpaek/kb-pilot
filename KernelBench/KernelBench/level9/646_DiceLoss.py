import torch
import torch.nn as nn
import torch.utils.data


def flatten_samples(input_):
    """
    Flattens a tensor or a variable such that the channel axis is first and the sample axis
    is second. The shapes are transformed as follows:
        (N, C, H, W) --> (C, N * H * W)
        (N, C, D, H, W) --> (C, N * D * H * W)
        (N, C) --> (C, N)
    The input must be atleast 2d.
    """
    num_channels = input_.size(1)
    permute_axes = list(range(input_.dim()))
    permute_axes[0], permute_axes[1] = permute_axes[1], permute_axes[0]
    permuted = input_.permute(*permute_axes).contiguous()
    flattened = permuted.view(num_channels, -1)
    return flattened


def dice_score(input_, target, invert=False, channelwise=True, eps=1e-07):
    if channelwise:
        input_ = flatten_samples(input_)
        target = flatten_samples(target)
        numerator = (input_ * target).sum(-1)
        denominator = (input_ * input_).sum(-1) + (target * target).sum(-1)
        channelwise_score = 2 * (numerator / denominator.clamp(min=eps))
        if invert:
            channelwise_score = 1.0 - channelwise_score
        score = channelwise_score.sum()
    else:
        numerator = (input_ * target).sum()
        denominator = (input_ * input_).sum() + (target * target).sum()
        score = 2.0 * (numerator / denominator.clamp(min=eps))
        if invert:
            score = 1.0 - score
    return score


class DiceLoss(nn.Module):

    def __init__(self, channelwise=True, eps=1e-07):
        super().__init__()
        self.channelwise = channelwise
        self.eps = eps
        self.init_kwargs = {'channelwise': channelwise, 'eps': self.eps}

    def forward(self, input_, target):
        return dice_score(input_, target, invert=True, channelwise=self.
            channelwise, eps=self.eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
