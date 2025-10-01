import torch
import torch.nn as nn


class PositioningCost(nn.Module):

    def __init__(self, target, Q=1, R=0, P=0):
        super().__init__()
        self.target = target
        self.Q, self.R, self.P = Q, R, P

    def forward(self, traj, u=None, mesh_p=None):
        cost = 0.1 * torch.norm(traj[..., -1, :3] - self.target
            ) + 1 * torch.norm(traj[..., :3] - self.target
            ) + 0.01 * torch.norm(traj[..., 3:6]) + 0.01 * torch.norm(traj[
            ..., 6:9]) + 0.01 * torch.norm(traj[..., 9:12])
        return cost


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'target': 4}]
