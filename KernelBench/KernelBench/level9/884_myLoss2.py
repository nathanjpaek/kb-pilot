import torch
import torch.nn.functional as F
import torch.nn as nn


class myLoss2(nn.Module):

    def __init__(self, alpha=1.0):
        super(myLoss2, self).__init__()
        self.alpha = alpha

    def forward(self, sent_probs, doc_probs, sent_targets, doc_targets):
        loss_1 = F.mse_loss(sent_probs, sent_targets)
        loss_2 = F.mse_loss(doc_probs, doc_targets)
        norm = 1.0 + self.alpha
        loss = (loss_1 + self.alpha * loss_2) / norm
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
