import torch
from torch import nn


class ThetaEncoder(nn.Module):

    def __init__(self, encoder_len):
        super(ThetaEncoder, self).__init__()
        self.encoder_len = encoder_len
        self.omega = 1

    def forward(self, theta):
        """
        :param theta: [B, lead_num, 2]
        :return: [B, lead_num, 12]
        """
        b, lead_num = theta.size(0), theta.size(1)
        sum_theta = theta[..., 0:1] + theta[..., 1:2]
        sub_theta = theta[..., 0:1] - theta[..., 1:2]
        before_encode = torch.cat([theta, sum_theta, sub_theta], dim=-1)
        out_all = [before_encode]
        for i in range(self.encoder_len):
            sin = torch.sin(before_encode * 2 ** i * self.omega)
            cos = torch.cos(before_encode * 2 ** i * self.omega)
            out_all += [sin, cos]
        after_encode = torch.stack(out_all, dim=-1).view(b, lead_num, -1)
        return after_encode


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'encoder_len': 4}]
