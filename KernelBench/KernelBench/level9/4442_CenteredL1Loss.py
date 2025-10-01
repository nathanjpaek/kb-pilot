import torch
from torch.nn.functional import l1_loss


class CenteredL1Loss(torch.nn.Module):

    def __init__(self, margin):
        super(CenteredL1Loss, self).__init__()
        self.m = margin

    def forward(self, true, preds):
        return l1_loss(preds[:, :, self.m:-self.m, self.m:-self.m], true[:,
            :, self.m:-self.m, self.m:-self.m])


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'margin': 4}]
