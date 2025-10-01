import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1.0 * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class triplet_my_loss(nn.Module):

    def __init__(self, margin=1.0, use_gpu=True):
        super(triplet_my_loss, self).__init__()
        self.use_gpu = use_gpu
        self.margin = margin
        self.mse = nn.MSELoss()

    def forward(self, inputs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        p = inputs[0]
        p1 = inputs[1]
        n1 = inputs[2]
        anchor = normalize(p, axis=-1)
        positive = normalize(p1, axis=-1)
        negative = normalize(n1, axis=-1)
        s1 = torch.sum(self.mse(anchor, positive))
        s2 = torch.sum(self.mse(anchor, negative))
        loss = torch.mul(torch.mul(s1, self.margin), torch.pow(s2, -1))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
