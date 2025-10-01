import torch
from torch import nn
from torch.autograd import Variable


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)
    return transposed.contiguous().view(C, -1)


def compute_per_channel_dice(input, target, epsilon=1e-05, ignore_index=
    None, weight=None):
    assert input.size() == target.size(
        ), "'input' and 'target' must have the same shape"
    target = target.float()
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
        sigmoid_normalization=True, skip_last_target=False):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        self.skip_last_target = skip_last_target

    def forward(self, input, target):
        input = self.normalization(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
        else:
            weight = None
        if self.skip_last_target:
            target = target[:, :-1, ...]
        per_channel_dice = compute_per_channel_dice(input, target, epsilon=
            self.epsilon, ignore_index=self.ignore_index, weight=weight)
        return torch.mean(1.0 - per_channel_dice)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
