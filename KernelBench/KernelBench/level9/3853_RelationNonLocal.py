import torch
import torch.nn as nn


class RelationNonLocal(nn.Module):

    def __init__(self, C):
        super(RelationNonLocal, self).__init__()
        self.conv_fv = nn.Conv2d(C, C, kernel_size=1, stride=1)
        self.conv_fk = nn.Conv2d(C, C, kernel_size=1, stride=1)
        self.conv_fq = nn.Conv2d(C, C, kernel_size=1, stride=1)
        self.conv_fr = nn.Conv2d(C, C, kernel_size=1, stride=1)

    def forward(self, input_):
        N, C, H, W = input_.shape
        f_v = self.conv_fv(input_)
        f_k = self.conv_fk(input_)
        f_q = self.conv_fq(input_)
        f_k = f_k.reshape([N, C, H * W]).permute(0, 2, 1)
        f_q = f_q.reshape([N, C, H * W])
        w = torch.matmul(f_k, f_q) / (H * W)
        f_r = torch.matmul(w.permute(0, 2, 1), f_v.reshape([N, C, H * W]).
            permute(0, 2, 1)).permute(0, 2, 1)
        f_r = f_r.reshape(N, C, H, W)
        f_r = self.conv_fr(f_r)
        return f_r


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'C': 4}]
