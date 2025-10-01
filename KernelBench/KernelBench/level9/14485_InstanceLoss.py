import torch
import torch.nn as nn
import torch.nn.init


class InstanceLoss(nn.Module):
    """
    Compute instance loss
    """

    def __init__(self):
        super(InstanceLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, img_cls, txt_cls, labels):
        cost_im = self.loss(img_cls, labels)
        cost_s = self.loss(txt_cls, labels)
        return cost_im + cost_s


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
