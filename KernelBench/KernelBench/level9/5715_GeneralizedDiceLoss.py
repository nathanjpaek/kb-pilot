import torch
from torch import nn
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
        return torch.zeros(shape).scatter(1, src.type(torch.LongTensor), 1)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)
    return transposed.view(C, -1)


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, epsilon=1e-05, weight=None, ignore_index=None,
        sigmoid_normalization=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target):
        input = self.normalization(input)
        if target.dim() == 4:
            target = expand_as_one_hot(target, C=input.size()[1],
                ignore_index=self.ignore_index)
        assert input.size() == target.size(
            ), "'input' and 'target' must have the same shape"
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False
            input = input * mask
            target = target * mask
        input = flatten(input)
        target = flatten(target)
        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1.0 / (target_sum * target_sum).clamp(min=
            self.epsilon), requires_grad=False)
        intersect = (input * target).sum(-1) * class_weights
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            intersect = weight * intersect
        denominator = (input + target).sum(-1) * class_weights
        return torch.mean(1.0 - 2.0 * intersect / denominator.clamp(min=
            self.epsilon))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
