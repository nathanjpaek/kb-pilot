import torch
import torch.utils.data
import torch.nn as nn


class Pointnet(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_dim, segmentation=
        False):
        super().__init__()
        self.fc_in = nn.Conv1d(in_channels, 2 * hidden_dim, 1)
        self.fc_0 = nn.Conv1d(2 * hidden_dim, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(2 * hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(2 * hidden_dim, hidden_dim, 1)
        self.fc_3 = nn.Conv1d(2 * hidden_dim, hidden_dim, 1)
        self.segmentation = segmentation
        if segmentation:
            self.fc_out = nn.Conv1d(2 * hidden_dim, out_channels, 1)
        else:
            self.fc_out = nn.Linear(hidden_dim, out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc_in(x)
        x = self.fc_0(self.activation(x))
        x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
        x = torch.cat([x, x_pool], dim=1)
        x = self.fc_1(self.activation(x))
        x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
        x = torch.cat([x, x_pool], dim=1)
        x = self.fc_2(self.activation(x))
        x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
        x = torch.cat([x, x_pool], dim=1)
        x = self.fc_3(self.activation(x))
        if self.segmentation:
            x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
            x = torch.cat([x, x_pool], dim=1)
        else:
            x = torch.max(x, dim=2)[0]
        x = self.fc_out(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'hidden_dim': 4}]
