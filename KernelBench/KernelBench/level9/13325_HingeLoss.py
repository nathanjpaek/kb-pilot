import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeLoss(nn.Module):
    """criterion for loss function
	y: 0/1 ground truth matrix of size: batch_size x output_size
	f: real number pred matrix of size: batch_size x output_size
	"""

    def __init__(self, margin=1.0, squared=True):
        super(HingeLoss, self).__init__()
        self.margin = margin
        self.squared = squared

    def forward(self, f, y, C_pos=1.0, C_neg=1.0):
        y_new = 2.0 * y - 1.0
        tmp = y_new * f
        loss = F.relu(self.margin - tmp)
        if self.squared:
            loss = loss ** 2
        loss = loss * (C_pos * y + C_neg * (1.0 - y))
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
