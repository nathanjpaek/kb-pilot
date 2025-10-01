import torch
import torch.utils.data
import torch
from torchvision.transforms import functional as F
from torch import nn
from torch.nn import functional as F


class AbstractConsistencyLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits1, logits2):
        raise NotImplementedError


class KLDivLossWithLogits(AbstractConsistencyLoss):

    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.kl_div_loss = nn.KLDivLoss(reduction=reduction)

    def forward(self, logits1, logits2):
        return self.kl_div_loss(F.log_softmax(logits1, dim=1), F.softmax(
            logits2, dim=1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
