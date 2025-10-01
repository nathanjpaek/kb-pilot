import torch
import torch.utils.data.distributed
import torch
import torch.nn as nn
from numpy import int64 as int64
from torchvision.transforms import functional as F
import torch.nn.functional as F
import torch.utils


class SoftCrossEntropyLoss2d(nn.Module):

    def __init__(self):
        super(SoftCrossEntropyLoss2d, self).__init__()

    def forward(self, inputs, targets):
        loss = 0
        inputs = -F.log_softmax(inputs, dim=1)
        for index in range(inputs.size()[0]):
            loss += F.conv2d(inputs[range(index, index + 1)], targets[range
                (index, index + 1)]) / (targets.size()[2] * targets.size()[3])
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
