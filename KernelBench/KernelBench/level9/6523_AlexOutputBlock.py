import torch
import torch.nn as nn
import torch.utils.data


class AlexDense(nn.Module):
    """
    AlexNet specific dense block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(AlexDense, self).__init__()
        self.fc = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.activ = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc(x)
        x = self.activ(x)
        x = self.dropout(x)
        return x


class AlexOutputBlock(nn.Module):
    """
    AlexNet specific output block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    classes : int
        Number of classification classes.
    """

    def __init__(self, in_channels, classes):
        super(AlexOutputBlock, self).__init__()
        mid_channels = 4096
        self.fc1 = AlexDense(in_channels=in_channels, out_channels=mid_channels
            )
        self.fc2 = AlexDense(in_channels=mid_channels, out_channels=
            mid_channels)
        self.fc3 = nn.Linear(in_features=mid_channels, out_features=classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'classes': 4}]
