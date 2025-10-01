import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.functional as F
import torch.cuda.amp


class DAImgHead(nn.Module):
    """
    Add a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAImgHead, self).__init__()
        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        for l in [self.conv1_da, self.conv2_da]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        feature = F.relu(self.conv1_da(x))
        feature = self.conv2_da(feature)
        return feature


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
