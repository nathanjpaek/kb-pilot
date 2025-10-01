import torch
import torch.nn as nn
import torch.utils.data


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-06

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class L1_Gradient_loss(nn.Module):

    def __init__(self):
        super(L1_Gradient_loss, self).__init__()
        self.eps = 1e-06
        self.crit = L1_Charbonnier_loss()

    def forward(self, X, Y):
        xgin = X[:, :, 1:, :] - X[:, :, 0:-1, :]
        ygin = X[:, :, :, 1:] - X[:, :, :, 0:-1]
        xgtarget = Y[:, :, 1:, :] - Y[:, :, 0:-1, :]
        ygtarget = Y[:, :, :, 1:] - Y[:, :, :, 0:-1]
        xl = self.crit(xgin, xgtarget)
        yl = self.crit(ygin, ygtarget)
        return (xl + yl) * 0.5


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
