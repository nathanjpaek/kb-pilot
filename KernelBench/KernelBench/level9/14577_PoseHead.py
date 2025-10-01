import torch
import torch.nn as nn


class PoseHead(nn.Module):

    def __init__(self, input_dim=256, hidden_dim=128):
        super(PoseHead, self).__init__()
        self.conv1_pose = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2_pose = nn.Conv2d(hidden_dim, 6, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_p):
        out = self.conv2_pose(self.relu(self.conv1_pose(x_p))).mean(3).mean(2)
        return torch.cat([out[:, :3], 0.01 * out[:, 3:]], dim=1)


def get_inputs():
    return [torch.rand([4, 256, 64, 64])]


def get_init_inputs():
    return [[], {}]
