import torch
from torchvision.transforms import *
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class KLDivLoss(nn.Module):

    def __init__(self):
        super(KLDivLoss, self).__init__()

    def forward(self, pred, label):
        T = 3
        predict = F.log_softmax(pred / T, dim=1)
        target_data = F.softmax(label / T, dim=1)
        target_data = target_data + 10 ** -7
        target = Variable(target_data.data, requires_grad=False)
        loss = T * T * ((target * (target.log() - predict)).sum(1).sum() /
            target.size()[0])
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
