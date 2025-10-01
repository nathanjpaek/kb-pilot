import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):

    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        loss = -torch.mean(F.logsigmoid(diff))
        return loss


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
