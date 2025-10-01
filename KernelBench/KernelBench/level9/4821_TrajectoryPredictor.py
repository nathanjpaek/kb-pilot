import torch
import torch.nn as nn


class TrajectoryPredictor(nn.Module):

    def __init__(self, pose_size, trajectory_size, hidden_size):
        super(TrajectoryPredictor, self).__init__()
        self.lp = nn.Linear(hidden_size, pose_size)
        self.fc = nn.Linear(pose_size + hidden_size, trajectory_size)

    def forward(self, x):
        pose_vector = self.lp(x)
        trajectory_vector = self.fc(torch.cat((pose_vector, x), dim=-1))
        mixed_vector = torch.cat((trajectory_vector, pose_vector), dim=-1)
        return mixed_vector


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'pose_size': 4, 'trajectory_size': 4, 'hidden_size': 4}]
