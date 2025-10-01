import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init


class KeyValueAttention(nn.Module):

    def __init__(self, query_size, key_size, value_size, hid_size, init_range):
        super(KeyValueAttention, self).__init__()
        self.key2hid = nn.Linear(key_size, hid_size)
        self.query2hid = nn.Linear(query_size, hid_size)
        self.hid2output = nn.Linear(hid_size, 1)
        self.key2hid.weight.data.uniform_(-init_range, init_range)
        self.key2hid.bias.data.fill_(0)
        self.query2hid.weight.data.uniform_(-init_range, init_range)
        self.query2hid.bias.data.fill_(0)
        self.hid2output.weight.data.uniform_(-init_range, init_range)
        self.hid2output.bias.data.fill_(0)

    def _bottle(self, linear, x):
        y = linear(x.view(-1, x.size(-1)))
        return y.view(x.size(0), x.size(1), -1)

    def forward_attn(self, h):
        logit = self.attn(h.view(-1, h.size(2))).view(h.size(0), h.size(1))
        return logit

    def forward(self, q, k, v, mask=None):
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()
        h_k = self._bottle(self.key2hid, k)
        h_q = self.query2hid(q)
        h = F.tanh(h_k + h_q.unsqueeze(1).expand_as(h_k))
        logit = self._bottle(self.hid2output, h).squeeze(2)
        logit = logit.sub(logit.max(1, keepdim=True)[0].expand_as(logit))
        if mask is not None:
            logit = torch.add(logit, Variable(mask))
        p = F.softmax(logit)
        w = p.unsqueeze(2).expand_as(v)
        h = torch.sum(torch.mul(v, w), 1, keepdim=True)
        h = h.transpose(0, 1).contiguous()
        return h, p


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'query_size': 4, 'key_size': 4, 'value_size': 4,
        'hid_size': 4, 'init_range': 4}]
