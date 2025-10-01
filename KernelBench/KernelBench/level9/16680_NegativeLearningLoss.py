import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.utils
import torch.distributed


class NegativeLearningLoss(nn.Module):

    def __init__(self, threshold=0.05):
        super(NegativeLearningLoss, self).__init__()
        self.threshold = threshold

    def forward(self, predict):
        mask = (predict < self.threshold).detach()
        negative_loss_item = -1 * mask * torch.log(1 - predict + 1e-06)
        negative_loss = torch.sum(negative_loss_item) / torch.sum(mask)
        return negative_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
