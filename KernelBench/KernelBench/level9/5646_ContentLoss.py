import torch
from torch import nn


class ContentLoss(nn.Module):
    """Module to compute the content loss. Allows arbitrary size style images
    during initialization and updating the content target.
    Usage: During loss network definition set compute_loss to False, to allow,
    after initialization iterate through ContentLoss modules and set
    compute_loss to True to perform the loss evaluation at every forward pass.
    When doing optimization for multiple content targets, perform a forward
    pass with the target images and then use update() to set the target to
    those images.
    """

    def __init__(self):
        super(ContentLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='sum')

    def forward(self, x, target):
        _, c, h, w = x.shape
        self.loss = self.criterion(x, target.detach()) / (c * h * w)
        return self.loss

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
