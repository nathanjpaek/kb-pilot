import torch
import torch.nn.functional as F


class SumLossModule(torch.nn.Module):

    def __init__(self):
        super(SumLossModule, self).__init__()

    def forward(self, predictions, targets):
        y_losses = F.cross_entropy(predictions, targets, reduction='none')
        y_losses = torch.sum(y_losses, dim=[1, 2])
        Y_loss = torch.logsumexp(y_losses, dim=0)
        return Y_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
