import torch
import torch.nn.functional as F
from torch import nn
from torch import einsum


class ConsensusAttention(nn.Module):

    def __init__(self, num_patches_side, attend_self=True,
        local_consensus_radius=0):
        super().__init__()
        self.attend_self = attend_self
        self.local_consensus_radius = local_consensus_radius
        if self.local_consensus_radius > 0:
            coors = torch.stack(torch.meshgrid(torch.arange(
                num_patches_side), torch.arange(num_patches_side))).float()
            coors = rearrange(coors, 'c h w -> (h w) c')
            dist = torch.cdist(coors, coors)
            mask_non_local = dist > self.local_consensus_radius
            mask_non_local = rearrange(mask_non_local, 'i j -> () i j')
            self.register_buffer('non_local_mask', mask_non_local)

    def forward(self, levels):
        _, n, _, d, device = *levels.shape, levels.device
        q, k, _v = levels, F.normalize(levels, dim=-1), levels
        sim = einsum('b i l        d, b j l d        -> b l i j', q, k
            ) * d ** -0.5
        if not self.attend_self:
            self_mask = torch.eye(n, device=device, dtype=torch.bool)
            self_mask = rearrange(self_mask, 'i j -> () () i j')
            sim.masked_fill_(self_mask, TOKEN_ATTEND_SELF_VALUE)
        if self.local_consensus_radius > 0:
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(self.non_local_mask, max_neg_value)
        attn = sim.softmax(dim=-1)
        out = einsum('b l i j, b j l d -> b i l d', attn, levels)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_patches_side': 4}]
