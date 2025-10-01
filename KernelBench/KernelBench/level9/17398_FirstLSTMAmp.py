import torch
import torch.utils.data
import torch.nn as nn


class FirstLSTMAmp(nn.Module):
    """
    First LSTM amplifier branch.

    Parameters:
    ----------
    in_features : int
        Number of input channels.
    out_features : int
        Number of output channels.
    """

    def __init__(self, in_features, out_features):
        super(FirstLSTMAmp, self).__init__()
        mid_features = in_features // 4
        self.fc1 = nn.Linear(in_features=in_features, out_features=mid_features
            )
        self.activ = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=mid_features, out_features=
            out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
