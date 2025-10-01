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


class LossWithLogits(AbstractConsistencyLoss):

    def __init__(self, reduction='mean', loss_cls=nn.L1Loss):
        super().__init__(reduction)
        self.loss_with_softmax = loss_cls(reduction=reduction)

    def forward(self, logits1, logits2):
        loss = self.loss_with_softmax(F.softmax(logits1, dim=1), F.softmax(
            logits2, dim=1))
        return loss


class DiscrepancyLossWithLogits(AbstractConsistencyLoss):

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
        self.loss = LossWithLogits(reduction=reduction, loss_cls=nn.L1Loss)

    def forward(self, logits1, logits2):
        return self.loss(logits1, logits2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
