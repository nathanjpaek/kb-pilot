import torch
from typing import Optional
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class KLDivLoss(_Loss):

    def __init__(self, size_average: 'Optional[bool]'=None, reduce:
        'Optional[bool]'=None, reduction: 'str'='mean') ->None:
        super(KLDivLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, inputs: 'torch.Tensor', targets: 'torch.Tensor'
        ) ->torch.Tensor:
        log_input = torch.log(inputs)
        return F.kl_div(log_input, targets, reduction=self.reduction)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
