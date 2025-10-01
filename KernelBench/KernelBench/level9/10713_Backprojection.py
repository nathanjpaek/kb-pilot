import torch
import torch.nn as nn


class Backprojection(nn.Module):

    def __init__(self, batch_size, height, width):
        super(Backprojection, self).__init__()
        self.N, self.H, self.W = batch_size, height, width
        yy, xx = torch.meshgrid([torch.arange(0.0, float(self.H)), torch.
            arange(0.0, float(self.W))])
        yy = yy.contiguous().view(-1)
        xx = xx.contiguous().view(-1)
        self.ones = nn.Parameter(torch.ones(self.N, 1, self.H * self.W),
            requires_grad=False)
        self.coord = torch.unsqueeze(torch.stack([xx, yy], 0), 0).repeat(self
            .N, 1, 1)
        self.coord = nn.Parameter(torch.cat([self.coord, self.ones], 1),
            requires_grad=False)

    def forward(self, depth, inv_K):
        cam_p_norm = torch.matmul(inv_K[:, :3, :3], self.coord[:depth.shape
            [0], :, :])
        cam_p_euc = depth.view(depth.shape[0], 1, -1) * cam_p_norm
        cam_p_h = torch.cat([cam_p_euc, self.ones[:depth.shape[0], :, :]], 1)
        return cam_p_h


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'batch_size': 4, 'height': 4, 'width': 4}]
