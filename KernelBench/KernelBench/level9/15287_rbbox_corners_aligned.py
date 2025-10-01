import torch
import torch.nn as nn


class rbbox_corners_aligned(nn.Module):

    def _init_(self, gboxes):
        super(rbbox_corners_aligned, self)._init_()
        self.corners_gboxes = gboxes
        return

    def forward(ctx, gboxes):
        N = gboxes.shape[0]
        center_x = gboxes[:, 0]
        center_y = gboxes[:, 1]
        x_d = gboxes[:, 2]
        y_d = gboxes[:, 3]
        corners = torch.zeros([N, 2, 4], device=gboxes.device, dtype=torch.
            float32)
        corners[:, 0, 0] = x_d.mul(-0.5)
        corners[:, 1, 0] = y_d.mul(-0.5)
        corners[:, 0, 1] = x_d.mul(-0.5)
        corners[:, 1, 1] = y_d.mul(0.5)
        corners[:, 0, 2] = x_d.mul(0.5)
        corners[:, 1, 2] = y_d.mul(0.5)
        corners[:, 0, 3] = x_d.mul(0.5)
        corners[:, 1, 3] = y_d.mul(-0.5)
        b = center_x.unsqueeze(1).repeat(1, 4).unsqueeze(1)
        c = center_y.unsqueeze(1).repeat(1, 4).unsqueeze(1)
        return corners + torch.cat((b, c), 1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
