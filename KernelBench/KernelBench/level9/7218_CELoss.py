import torch
from torch import nn


class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        self.criterionBinary = nn.BCELoss(size_average=True)
        self.criterionMulti = nn.NLLLoss(size_average=True)

    def __repr__(self):
        return 'CE'

    def forward(self, output, target):
        if target.shape[1] == 1:
            loss = self.criterionBinary(output, target)
        else:
            target = torch.argmax(target, dim=1).long()
            loss = self.criterionMulti(torch.log(output), target)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
