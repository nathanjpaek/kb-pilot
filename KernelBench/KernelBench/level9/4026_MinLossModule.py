import torch
import torch.nn.functional as F


class MinLossModule(torch.nn.Module):

    def __init__(self):
        super(MinLossModule, self).__init__()

    def forward(self, predictions, targets):
        y_losses = F.cross_entropy(predictions, targets, reduction='none')
        y_losses = torch.sum(y_losses, dim=[1, 2])
        Y_loss = torch.min(y_losses)
        return Y_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
