import torch
import torch.nn.functional as F
import torch.nn as nn


class ConvModel(nn.Module):
    """Convolution 2D model."""

    def __init__(self, input_dim, output_dim):
        super(ConvModel, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=2,
            padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=
            2, padding=(1, 1))
        self.fc1 = nn.Linear(in_features=16 * (input_dim[0] + 2) * (
            input_dim[1] + 2), out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1)))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': [4, 4], 'output_dim': 4}]
