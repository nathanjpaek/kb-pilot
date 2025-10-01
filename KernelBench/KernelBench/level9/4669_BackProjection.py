import torch
import torch.nn as nn
import torch.utils.data


class BackProjection(nn.Module):
    """
        forward method:
            bbox3d: [N, 7] homo_x, homo_y, z, w, h, l, alpha
            p2: [3, 4]
            return [x3d, y3d, z, w, h, l, alpha]
    """

    def forward(self, bbox3d, p2):
        """
            bbox3d: [N, 7] homo_x, homo_y, z, w, h, l, alpha
            p2: [3, 4]
            return [x3d, y3d, z, w, h, l, alpha]
        """
        fx = p2[0, 0]
        fy = p2[1, 1]
        cx = p2[0, 2]
        cy = p2[1, 2]
        tx = p2[0, 3]
        ty = p2[1, 3]
        z3d = bbox3d[:, 2:3]
        x3d = (bbox3d[:, 0:1] * z3d - cx * z3d - tx) / fx
        y3d = (bbox3d[:, 1:2] * z3d - cy * z3d - ty) / fy
        return torch.cat([x3d, y3d, bbox3d[:, 2:]], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
