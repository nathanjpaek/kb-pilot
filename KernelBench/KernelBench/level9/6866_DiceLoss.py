import torch
from torch import nn as nn
from torch.autograd import Variable


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    assert input.dim() == 4
    shape = input.size()
    shape = list(shape)
    shape.insert(1, C)
    shape = tuple(shape)
    src = input.unsqueeze(0)
    if ignore_index is not None:
        expanded_src = src.expand(shape)
        mask = expanded_src == ignore_index
        src = src.clone()
        src[src == ignore_index] = 0
        result = torch.zeros(shape).scatter_(1, src, 1)
        result[mask] = ignore_index
        return result
    else:
        return torch.zeros(shape).scatter_(1, src, 1)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)
    return transposed.view(C, -1)


def compute_per_channel_dice(input, target, epsilon=1e-05, ignore_index=
    None, weight=None):
    if target.dim() == 4:
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=
            ignore_index)
    assert input.size() == target.size(
        ), "'input' and 'target' must have the same shape"
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False
        input = input * mask
        target = target * mask
    input = flatten(input)
    target = flatten(target)
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect
    denominator = (input + target).sum(-1)
    return 2.0 * intersect / denominator.clamp(min=epsilon)


class DiceLoss(nn.Module):
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    Additionally allows per-class weights to be provided.
    """

    def __init__(self, epsilon=1e-05, weight=None, ignore_index=None,
        sigmoid_normalization=True):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target):
        input = self.normalization(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
        else:
            weight = None
        per_channel_dice = compute_per_channel_dice(input, target, epsilon=
            self.epsilon, ignore_index=self.ignore_index, weight=weight)
        return torch.mean(1.0 - per_channel_dice)


def get_inputs():
    return [torch.ones([4, 4], dtype=torch.int64), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
