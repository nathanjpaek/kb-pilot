import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""

    def __init__(self, feature_dim, queue_size, temperature=0.07, thresh=0):
        super(MemoryMoCo, self).__init__()
        self.queue_size = queue_size
        self.temperature = temperature
        self.index = 0
        self.thresh = thresh
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1.0 / math.sqrt(feature_dim / 3)
        memory = torch.rand(self.queue_size, feature_dim, requires_grad=False
            ).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)

    def forward(self, q, k, k_all, update=False):
        k = k.detach()
        l_pos = (q * k).sum(dim=-1, keepdim=True)
        l_neg = torch.mm(q, self.memory.clone().detach().t())
        if self.thresh != 0:
            prob_pos1 = l_pos
            prob_pos_hard1 = prob_pos1 - self.thresh
            N = l_pos.size(0)
            Q = l_neg.size(1)
            prob_pos_hard1 = torch.add(l_neg, -1, prob_pos_hard1.expand(N, Q))
            l_neg[prob_pos_hard1 < 0] = 0
            l_neg[l_neg <= 0].size(0) / N / Q
            None
        out = torch.cat((l_pos, l_neg), dim=1)
        out = torch.div(out, self.temperature).contiguous()
        if update:
            with torch.no_grad():
                all_size = k_all.shape[0]
                out_ids = torch.fmod(torch.arange(all_size, dtype=torch.
                    long) + self.index, self.queue_size)
                self.memory.index_copy_(0, out_ids, k_all)
                self.index = (self.index + all_size) % self.queue_size
        return out


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'feature_dim': 4, 'queue_size': 4}]
