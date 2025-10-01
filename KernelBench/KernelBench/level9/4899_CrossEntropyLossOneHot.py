import torch
from torch import nn


class CrossEntropyLossOneHot(nn.Module):

    def __init__(self):
        super(CrossEntropyLossOneHot, self).__init__()
        self.soft_max = nn.LogSoftmax(dim=-1)
        self.nll_loss = nn.NLLLoss()

    def forward(self, preds, labels):
        """
        preds: [batch_size, label_size]
        labels: [batch_size, label_size] - One hot encoding by ground truth
        """
        batch_size = preds.shape[0]
        soft_preds = self.soft_max(preds)
        mul_res = torch.mul(soft_preds, labels)
        sum_res = torch.sum(-mul_res, dim=-1)
        cross_entropy_loss = torch.sum(sum_res, dim=0) / batch_size
        return cross_entropy_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
