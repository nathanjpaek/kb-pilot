import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(logits, labels):
    return torch.mean((1 - labels) * logits + torch.log(1 + torch.exp(-logits))
        )


class EdgeLoss(nn.Module):

    def __init__(self):
        super().__init__()
        laplace = torch.FloatTensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
            ).view([1, 1, 3, 3])
        self.laplace = nn.Parameter(data=laplace, requires_grad=False)

    def torchLaplace(self, x):
        edge = F.conv2d(x, self.laplace, padding=1)
        edge = torch.abs(torch.tanh(edge))
        return edge

    def forward(self, y_pred, y_true, mode=None):
        y_true_edge = self.torchLaplace(y_true)
        y_pred_edge = self.torchLaplace(y_pred)
        edge_loss = cross_entropy(y_pred_edge, y_true_edge)
        return edge_loss


def get_inputs():
    return [torch.rand([4, 1, 64, 64]), torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
