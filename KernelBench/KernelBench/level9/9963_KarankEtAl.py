import torch
import torch.nn as nn
import torch.nn.functional as F


class KarankEtAl(nn.Module):

    def __init__(self, input_channels, n_classes, patch_size=5):
        super(KarankEtAl, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.conv1 = nn.Conv3d(1, 3 * self.input_channels, (1, 3, 3))
        self.conv2 = nn.Conv3d(3 * self.input_channels, 9 * self.
            input_channels, (1, 3, 3))
        self.features_size = self._get_final_flattened_size()
        self.fc1 = nn.Linear(self.features_size, 6 * self.input_channels)
        self.fc2 = nn.Linear(6 * self.input_channels, self.n_classes)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels, self.patch_size,
                self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64, 64])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'n_classes': 4}]
