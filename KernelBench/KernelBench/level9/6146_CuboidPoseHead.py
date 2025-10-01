import torch
import torch.nn as nn
import torch.nn.functional as F


class CuboidPoseHead(nn.Module):

    def __init__(self, beta):
        """Get results from the 3D human pose heatmap. Instead of obtaining
        maximums on the heatmap, this module regresses the coordinates of
        keypoints via integral pose regression. Refer to `paper.

        <https://arxiv.org/abs/2004.06239>` for more details.

        Args:
            beta: Constant to adjust the magnification of soft-maxed heatmap.
        """
        super(CuboidPoseHead, self).__init__()
        self.beta = beta
        self.loss = nn.L1Loss()

    def forward(self, heatmap_volumes, grid_coordinates):
        """

        Args:
            heatmap_volumes (torch.Tensor(NxKxLxWxH)):
                3D human pose heatmaps predicted by the network.
            grid_coordinates (torch.Tensor(Nx(LxWxH)x3)):
                Coordinates of the grids in the heatmap volumes.
        Returns:
            human_poses (torch.Tensor(NxKx3)): Coordinates of human poses.
        """
        batch_size = heatmap_volumes.size(0)
        channel = heatmap_volumes.size(1)
        x = heatmap_volumes.reshape(batch_size, channel, -1, 1)
        x = F.softmax(self.beta * x, dim=2)
        grid_coordinates = grid_coordinates.unsqueeze(1)
        x = torch.mul(x, grid_coordinates)
        human_poses = torch.sum(x, dim=2)
        return human_poses

    def get_loss(self, preds, targets, weights):
        return dict(loss_pose=self.loss(preds * weights, targets * weights))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'beta': 4}]
