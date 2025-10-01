import torch
import torch.nn as nn
import torch.nn.functional as Func


class CrossAttN_v8(nn.Module):

    def __init__(self, in_planes, clip_dim):
        super(CrossAttN_v8, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, 1, 1, 0)
        self.g = nn.Conv2d(in_planes, in_planes, 1, 1, 0)
        self.h = nn.Conv2d(in_planes, in_planes, 1, 1, 0)
        self.output = nn.Conv2d(in_planes, in_planes, 1, 1, 0)
        return

    def forward(self, F_c, F_s):
        b, _c = F_c.shape[0], F_c.shape[1]
        F = self.f(F_c)
        F = F.view(b, F.shape[1], -1)
        G = self.g(F_s)
        G = G.view(b, G.shape[1], -1)
        H = self.h(F_s)
        H = H.view(b, H.shape[1], -1)
        S = torch.bmm(F.permute(0, 2, 1), G)
        S = Func.softmax(S, dim=-1)
        result = torch.bmm(H, S.permute(0, 2, 1))
        result = result.view(b, result.shape[1], F_c.shape[2], F_c.shape[3])
        result = self.output(result)
        return result


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_planes': 4, 'clip_dim': 4}]
