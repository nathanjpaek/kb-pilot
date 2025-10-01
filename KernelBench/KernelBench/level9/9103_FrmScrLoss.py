import torch
import torch.nn as nn


class FrmScrLoss(nn.Module):

    def __init__(self, propotion):
        super().__init__()
        self.s = propotion

    def forward(self, frm_scrs, label):
        _n, t, _c = frm_scrs.size()
        max_frm_values, _ = torch.topk(frm_scrs, max(int(t // self.s), 1), 1)
        mean_max_frm = max_frm_values.mean(1)
        min_frm_values, _ = torch.topk(-frm_scrs, max(int(t // self.s), 1), 1)
        mean_min_frm = -min_frm_values.mean(1)
        temporal_loss = (mean_min_frm - mean_max_frm) * label
        temporal_loss = temporal_loss.sum(-1).mean(-1)
        frm_scrs = frm_scrs * label[:, None, :]
        frm_act_scrs = frm_scrs[..., :-1]
        frm_bck_scr = frm_scrs[..., -1]
        frm_act_scr = frm_act_scrs.max(-1)[0]
        categorcial_loss = -1.0 * torch.abs(frm_act_scr - frm_bck_scr).mean(-1
            ).mean(-1)
        return temporal_loss + categorcial_loss


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'propotion': 4}]
