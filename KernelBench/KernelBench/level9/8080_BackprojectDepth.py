import torch
import torch.nn as nn


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height *
            self.width), requires_grad=False)

    def forward(self, depth, norm_pix_coords):
        cam_points = depth * norm_pix_coords
        cam_points = torch.cat([cam_points.view(self.batch_size, cam_points
            .shape[1], -1), self.ones], 1)
        return cam_points


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'batch_size': 4, 'height': 4, 'width': 4}]
