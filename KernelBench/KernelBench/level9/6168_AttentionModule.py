import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModule(nn.Module):
    """ A neural module that takes a feature map, attends to the features, and
    produces an attention.
    """

    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(dim, 1, kernel_size=1, padding=0)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        self.dim = dim

    def forward(self, feats):
        out = F.relu(self.conv1(feats))
        out = F.relu(self.conv2(out))
        out = torch.sigmoid(self.conv3(out))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
