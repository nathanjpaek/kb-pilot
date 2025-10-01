import math
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data.dataset


class MemoryReader(torch.nn.Module):

    def __init__(self):
        super(MemoryReader, self).__init__()

    def forward(self, m_key, m_val, q_key, q_val):
        B, D_e, T, H, W = m_key.size()
        _, D_o, _, _, _ = m_val.size()
        mi = m_key.view(B, D_e, T * H * W)
        mi = torch.transpose(mi, 1, 2)
        qi = q_key.view(B, D_e, H * W)
        p = torch.bmm(mi, qi)
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1)
        mo = m_val.view(B, D_o, T * H * W)
        mem = torch.bmm(mo, p)
        mem = mem.view(B, D_o, H, W)
        mem_val = torch.cat([mem, q_val], dim=1)
        return mem_val, p


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4, 4]), torch
        .rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
