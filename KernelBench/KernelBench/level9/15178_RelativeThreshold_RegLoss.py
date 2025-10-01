import torch
import torch.nn as nn
import torch.nn.init


class RelativeThreshold_RegLoss(nn.Module):

    def __init__(self, threshold, size_average=True):
        super(RelativeThreshold_RegLoss, self).__init__()
        self.size_average = size_average
        self.eps = 1e-07
        self.threshold = threshold

    def forward(self, preds, targets):
        """
            Args:
                inputs:(n, h, w, d)
                targets:(n, h, w, d)  
        """
        assert not targets.requires_grad
        assert preds.shape == targets.shape, 'dim of preds and targets are different'
        dist = torch.abs(preds - targets).view(-1)
        baseV = targets.view(-1)
        baseV = torch.abs(baseV + self.eps)
        relativeDist = torch.div(dist, baseV)
        mask = relativeDist.ge(self.threshold)
        largerLossVec = torch.masked_select(dist, mask)
        loss = torch.mean(largerLossVec)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'threshold': 4}]
