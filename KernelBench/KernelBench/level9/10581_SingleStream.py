from _paritybench_helpers import _mock_config
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as torch_init


def calculate_l1_norm(f):
    f_norm = torch.norm(f, p=2, dim=-1, keepdim=True)
    f = torch.div(f, f_norm)
    return f


class SingleStream(nn.Module):

    def __init__(self, args, n_out):
        super().__init__()
        self.n_out = n_out
        self.n_class = args.class_num
        self.scale_factor = args.scale_factor
        self.ac_center = nn.Parameter(torch.zeros(self.n_class + 1, self.n_out)
            )
        torch_init.xavier_uniform_(self.ac_center)
        self.fg_center = nn.Parameter(-1.0 * self.ac_center[-1, ...][None, ...]
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_emb):
        norms_emb = calculate_l1_norm(x_emb)
        norms_ac = calculate_l1_norm(self.ac_center)
        norms_fg = calculate_l1_norm(self.fg_center)
        frm_scrs = torch.einsum('ntd,cd->ntc', [norms_emb, norms_ac]
            ) * self.scale_factor
        frm_fb_scrs = torch.einsum('ntd,kd->ntk', [norms_emb, norms_fg]
            ).squeeze(-1) * self.scale_factor
        class_agno_att = self.sigmoid(frm_fb_scrs)
        class_wise_att = self.sigmoid(frm_scrs)
        class_agno_norm_att = class_agno_att / torch.sum(class_agno_att,
            dim=1, keepdim=True)
        class_wise_norm_att = class_wise_att / torch.sum(class_wise_att,
            dim=1, keepdim=True)
        ca_vid_feat = torch.einsum('ntd,nt->nd', [x_emb, class_agno_norm_att])
        ca_vid_norm_feat = calculate_l1_norm(ca_vid_feat)
        ca_vid_scr = torch.einsum('nd,cd->nc', [ca_vid_norm_feat, norms_ac]
            ) * self.scale_factor
        ca_vid_pred = F.softmax(ca_vid_scr, -1)
        mil_vid_feat = torch.einsum('ntd,ntc->ncd', [x_emb,
            class_wise_norm_att])
        mil_vid_norm_feat = calculate_l1_norm(mil_vid_feat)
        mil_vid_scr = torch.einsum('ncd,cd->nc', [mil_vid_norm_feat, norms_ac]
            ) * self.scale_factor
        mil_vid_pred = F.softmax(mil_vid_scr, -1)
        return ca_vid_pred, mil_vid_pred, class_agno_att, frm_scrs


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(class_num=4, scale_factor=1.0),
        'n_out': 4}]
