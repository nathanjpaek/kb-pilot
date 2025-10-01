import torch
import torch.nn as nn
import torch.utils.data


class L2Loss(nn.Module):
    """
  Compute the l2 distance
  """

    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, h_pred, h_target):
        return torch.norm(h_target - h_pred, p=2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
