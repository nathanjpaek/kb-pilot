from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import torch.nn.init


class ImgAttention(nn.Module):

    def __init__(self, opt):
        super(ImgAttention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats):
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        att_h = self.h2att(h)
        att_h = att_h.unsqueeze(1).expand_as(att)
        dot = att + att_h
        dot = F.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)
        dot = self.alpha_net(dot)
        dot = dot.view(-1, att_size)
        weight = F.softmax(dot)
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)
        return att_res


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4,
        4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config(rnn_size=4, att_hid_size=4)}]
