import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvElement(nn.Module):
    """
    Residual Core element used inside the NN. Control the number of filters
    and batch normalization.
    """

    def __init__(self, input_size, num_filters, use_leaky=True, stride=1,
        leaky_p=0.2):
        super(ConvElement, self).__init__()
        self.use_lr = use_leaky
        self.leaky_p = leaky_p
        self.conv1 = nn.Conv3d(input_size, num_filters, kernel_size=3,
            padding=1, stride=stride)

    def forward(self, x):
        """
        include residual model
        """
        x_1 = self.conv1(x)
        return F.leaky_relu(x_1, self.leaky_p) if self.use_lr else F.relu(x_1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'num_filters': 4}]
