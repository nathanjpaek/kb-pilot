import torch
from torch.nn import functional as F
import torch.nn as nn


def cal_normal(v, dim=-1, keepdim=False):
    """

    :return:
    """
    normal = torch.sum(v ** 2, dim=dim, keepdim=keepdim) ** 0.5
    return normal


def squash(sr, dim=1):
    """

    :param dim:
    :param sr:(bs, dim)
    :return:
    """
    sr_normal = cal_normal(sr, keepdim=True, dim=dim)
    sr_normal2 = sr_normal ** 2
    v = sr / sr_normal * (sr_normal2 / (1 + sr_normal2))
    return v


def dynamic_routing(u, br):
    """
    u: (b, num_size, num_classes, dim)
    br: (b, num_size, num_classes, 1)
    :return:
    """
    cr = F.softmax(br, dim=1)
    sr = torch.sum(cr * u, dim=1)
    vr = squash(sr, dim=-1)
    sm = torch.einsum('bncd,bcd->bnc', u, vr).unsqueeze(dim=3)
    br = br + sm
    return br, vr


class RoutingBase(nn.Module):

    def __init__(self, num_routing_iterations=1, **kwargs):
        super(RoutingBase, self).__init__()
        self.num_routing_iterations = num_routing_iterations

    def forward(self, inx):
        """
        inx: (b, num_size, num_classes, dim)
        :return:
        """
        v_h = []
        b_h = []
        inx_device = inx.device
        br = torch.zeros(size=(*inx.size()[:-1], 1), requires_grad=False,
            device=inx_device)
        for i in range(self.num_routing_iterations):
            br, vr = dynamic_routing(inx, br)
            v_h.append(vr.unsqueeze(dim=3))
            b_h.append(br)
        return torch.cat(b_h, dim=-1), torch.cat(v_h, dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
