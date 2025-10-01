from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
from torch.autograd import *
import torch.nn.functional as F


class SingleSP(nn.Module):

    def __init__(self, opt):
        super(SingleSP, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.wh = nn.Linear(self.rnn_size, self.att_hid_size)
        self.wa = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, roi_feats, p_roi_feats, att_masks=None):
        dot = self.wh(h).unsqueeze(1).expand_as(p_roi_feats) + p_roi_feats
        weight = F.softmax(self.wa(torch.tanh(dot)).squeeze(2), dim=1)
        if att_masks is not None:
            weight = weight * att_masks
            weight = weight / weight.sum(1, keepdim=True)
        single_feat = torch.bmm(weight.unsqueeze(1), roi_feats).squeeze(1)
        return single_feat


def get_inputs():
    return [torch.rand([4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config(rnn_size=4, att_hid_size=4)}]
