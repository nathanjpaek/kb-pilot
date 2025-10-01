import torch
import torch.nn as nn


class mbr_convex_hull(nn.Module):
    """
        Miminum Bounding Rectangle (MBR)
        Algorithm core: The orientation of the MBR is the same as the one of one of the edges of the point cloud convex hull, which means
        the result rectangle must overlap with at least one of the edges.
    """

    def _init_(self, hull_points_2d):
        super(mbr_convex_hull, self)._init_()
        self.hull_points_2d = hull_points_2d
        return

    def forward(ctx, hull_points_2d):
        N = hull_points_2d.shape[0]
        edges = hull_points_2d[1:N, :].add(-hull_points_2d[0:N - 1, :])
        edge_angles = torch.atan2(edges[:, 1], edges[:, 0])
        edge_angles = torch.fmod(edge_angles, 3.1415926 / 2.0)
        edge_angles = torch.abs(edge_angles)
        a = torch.stack((torch.cos(edge_angles), torch.cos(edge_angles - 
            3.1415926 / 2.0)), 1)
        a = torch.unsqueeze(a, 1)
        b = torch.stack((torch.cos(edge_angles + 3.1415926 / 2.0), torch.
            cos(edge_angles)), 1)
        b = torch.unsqueeze(b, 1)
        R_tensor = torch.cat((a, b), 1)
        hull_points_2d_ = torch.unsqueeze(torch.transpose(hull_points_2d, 0,
            1), 0)
        rot_points = R_tensor.matmul(hull_points_2d_)
        min_x = torch.min(rot_points, 2)[0]
        max_x = torch.max(rot_points, 2)[0]
        areas = (max_x[:, 0] - min_x[:, 0]).mul(max_x[:, 1] - min_x[:, 1])
        return torch.min(areas)


def get_inputs():
    return [torch.rand([4, 2, 4])]


def get_init_inputs():
    return [[], {}]
