import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import torch.utils.data


class LinearI_Neg(nn.Linear):

    def forward(self, x):
        return F.linear(x, -self.weight.exp(), self.bias)

    def ibp_forward(self, l, u):
        weight = -self.weight.exp()
        if self.bias is not None:
            l_ = (weight.clamp(min=0) @ l.t() + weight.clamp(max=0) @ u.t() +
                self.bias[:, None]).t()
            u_ = (weight.clamp(min=0) @ u.t() + weight.clamp(max=0) @ l.t() +
                self.bias[:, None]).t()
        else:
            l_ = (weight.clamp(min=0) @ l.t() + weight.clamp(max=0) @ u.t()).t(
                )
            u_ = (weight.clamp(min=0) @ u.t() + weight.clamp(max=0) @ l.t()).t(
                )
        return l_, u_


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
