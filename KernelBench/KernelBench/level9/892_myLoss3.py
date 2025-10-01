import torch
import torch.nn.functional as F
import torch.nn as nn


class myLoss3(nn.Module):

    def __init__(self, alpha=1.0, beta=1.0):
        super(myLoss3, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, sent_probs, doc_probs, event_probs, sent_targets,
        doc_targets, event_targets):
        loss_1 = F.mse_loss(sent_probs, sent_targets)
        loss_2 = F.mse_loss(doc_probs, doc_targets)
        loss_3 = F.mse_loss(event_probs, event_targets)
        norm = 1.0 + self.alpha + self.beta
        loss = (loss_1 + self.alpha * loss_2 + self.beta * loss_3) / norm
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]),
        torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
