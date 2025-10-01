import torch
from torch.nn import init as init
from torch.nn.modules.loss import _Loss


class L1_Charbonnier_loss_color(_Loss):
    """
    L1 Charbonnierloss color
    """

    def __init__(self, para):
        super(L1_Charbonnier_loss_color, self).__init__()
        self.eps = 0.001

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        diff_sq = diff * diff
        diff_sq_color = torch.mean(diff_sq, 1, True)
        error = torch.sqrt(diff_sq_color + self.eps * self.eps)
        loss = torch.mean(error)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'para': 4}]
