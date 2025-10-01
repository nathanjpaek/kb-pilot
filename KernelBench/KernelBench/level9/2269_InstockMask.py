import torch
import torch.nn as nn


class InstockMask(nn.Module):

    def __init__(self, time_step, ltsp, min_instock_ratio=0.5,
        eps_instock_dph=0.001, eps_total_dph=0.001, **kwargs):
        super(InstockMask, self).__init__(**kwargs)
        if not eps_total_dph > 0:
            raise ValueError(
                f'epsilon_total_dph of {eps_total_dph} is invalid!                               This parameter must be > 0 to avoid division by 0.'
                )
        self.min_instock_ratio = min_instock_ratio
        self.eps_instock_dph = eps_instock_dph
        self.eps_total_dph = eps_total_dph

    def forward(self, demand, total_dph, instock_dph):
        if total_dph is not None and instock_dph is not None:
            total_dph = total_dph + self.eps_total_dph
            instock_dph = instock_dph + self.eps_instock_dph
            instock_rate = torch.round(instock_dph / total_dph)
            demand = torch.where(instock_rate >= self.min_instock_ratio,
                demand, -torch.ones_like(demand))
        return demand


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'time_step': 4, 'ltsp': 4}]
