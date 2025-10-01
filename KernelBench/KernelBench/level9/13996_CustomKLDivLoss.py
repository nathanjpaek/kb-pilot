import torch
from torch import Tensor
from torch.nn import functional as F


class CustomKLDivLoss(torch.nn.Module):

    def __init__(self, reduction='batchmean', log_target=False,
        apply_softmax=True) ->None:
        super(CustomKLDivLoss, self).__init__()
        self.reduction = reduction
        self.log_target = log_target
        self.apply_softmax = apply_softmax

    def forward(self, input: 'Tensor', target: 'Tensor') ->Tensor:
        if self.apply_softmax:
            target = torch.softmax(target, dim=-1)
        return F.kl_div(torch.log_softmax(input, dim=-1), target, reduction
            =self.reduction, log_target=self.log_target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
