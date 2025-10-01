import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.parallel
import torch.optim
import torch.utils.data


def binary_cross_entropy(input, target, eps=1e-10):
    """if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}) is deprecated. "
                      "Please ensure they have the same size.".format(target.size(), input.size()))
    if input.nelement() != target.nelement():
        raise ValueError("Target and input must have the same number of elements. target nelement ({}) "
                         "!= input nelement ({})".format(target.nelement(), input.nelement()))

    if weight is not None:
        new_size = _infer_size(target.size(), weight.size())
        weight = weight.expand(new_size)
        if torch.is_tensor(weight):
            weight = Variable(weight)"""
    input = torch.sigmoid(input)
    return -(target * torch.log(input + eps) + (1 - target) * torch.log(1 -
        input + eps))


class MultiLabelSoftMarginLoss(_WeightedLoss):

    def forward(self, input, target):
        return binary_cross_entropy(input, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
