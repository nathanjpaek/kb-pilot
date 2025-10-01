import torch
import torch.nn as nn


class OffsetNet(nn.Module):
    """OffsetNet in Temporal interlace module.

    The OffsetNet consists of one convolution layer and two fc layers
    with a relu activation following with a sigmoid function. Following
    the convolution layer, two fc layers and relu are applied to the output.
    Then, apply the sigmoid function with a multiply factor and a minus 0.5
    to transform the output to (-4, 4).

    Args:
        in_channels (int): Channel num of input features.
        groups (int): Number of groups for fc layer outputs.
        num_segments (int): Number of frame segments.
    """

    def __init__(self, in_channels, groups, num_segments):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        kernel_size = 3
        padding = 1
        self.conv = nn.Conv1d(in_channels, 1, kernel_size, padding=padding)
        self.fc1 = nn.Linear(num_segments, num_segments)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_segments, groups)
        self.init_weights()

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        self.fc2.bias.data[...] = 0.5108

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        n, _, t = x.shape
        x = self.conv(x)
        x = x.view(n, t)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(n, 1, -1)
        x = 4 * (self.sigmoid(x) - 0.5)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'groups': 1, 'num_segments': 4}]
