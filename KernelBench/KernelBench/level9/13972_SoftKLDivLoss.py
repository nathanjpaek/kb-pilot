import torch
from torch import Tensor
from torch.nn import functional as F


class SoftKLDivLoss(torch.nn.Module):

    def __init__(self, temp=20.0, reduction='batchmean', log_target=False
        ) ->None:
        super(SoftKLDivLoss, self).__init__()
        self.temp = temp
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, input: 'Tensor', target: 'Tensor') ->Tensor:
        soft_input = torch.log_softmax(input / self.temp, dim=-1)
        soft_target = torch.softmax(target / self.temp, dim=-1)
        return self.temp ** 2 * F.kl_div(soft_input, soft_target, reduction
            =self.reduction, log_target=self.log_target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
