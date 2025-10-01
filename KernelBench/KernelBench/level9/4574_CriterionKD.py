import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._utils
import torch.optim


class CriterionKD(nn.Module):
    """
    knowledge distillation loss
    """

    def __init__(self, upsample=False, temperature=4):
        super(CriterionKD, self).__init__()
        self.upsample = upsample
        self.temperature = temperature
        self.criterion_kd = torch.nn.KLDivLoss()

    def forward(self, pred, soft):
        soft.detach()
        h, w = soft.size(2), soft.size(3)
        if self.upsample:
            scale_pred = F.interpolate(input=pred, size=(h * 2, w * 2),
                mode='bilinear', align_corners=True)
            scale_soft = F.interpolate(input=soft, size=(h * 2, w * 2),
                mode='bilinear', align_corners=True)
        else:
            scale_pred = pred
            scale_soft = soft
        loss = self.criterion_kd(F.log_softmax(scale_pred / self.
            temperature, dim=1), F.softmax(scale_soft / self.temperature,
            dim=1))
        return loss * self.temperature * self.temperature


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
