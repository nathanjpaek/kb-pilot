import torch
from torch import nn
import torch.utils.data


class NCESoftmaxLoss(nn.Module):

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, label):
        x.shape[0]
        x = x.squeeze()
        loss = self.criterion(x, label)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
