from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
from torch.autograd import *
import torch.nn.functional as F


class OutputSP(nn.Module):

    def __init__(self, opt):
        super(OutputSP, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.wv = nn.Linear(self.rnn_size, self.att_hid_size)
        self.wh = nn.Linear(self.rnn_size, self.att_hid_size)
        self.wa = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, single_feat, comp_feat, fc_feats):
        feats = torch.stack([single_feat, comp_feat, fc_feats], dim=1)
        feats_ = self.wv(feats)
        dot = self.wh(h).unsqueeze(1).expand_as(feats_) + feats_
        weight = F.softmax(self.wa(torch.tanh(dot)).squeeze(2), dim=1)
        output_feat = torch.bmm(weight.unsqueeze(1), feats).squeeze(1)
        return output_feat


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]),
        torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config(rnn_size=4, att_hid_size=4)}]
