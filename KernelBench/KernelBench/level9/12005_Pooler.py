import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed


class Pooler(nn.Module):
    """ Do pooling, possibly with a projection beforehand """

    def __init__(self, d_inp, project=True, d_proj=512, pool_type='max'):
        super(Pooler, self).__init__()
        self.project = nn.Linear(d_inp, d_proj) if project else lambda x: x
        self.pool_type = pool_type

    def forward(self, sequence, mask):
        if len(mask.size()) < 3:
            mask = mask.unsqueeze(dim=-1)
        pad_mask = mask == 0
        proj_seq = self.project(sequence)
        if self.pool_type == 'max':
            proj_seq = proj_seq.masked_fill(pad_mask, -float('inf'))
            seq_emb = proj_seq.max(dim=1)[0]
        elif self.pool_type == 'mean':
            proj_seq = proj_seq.masked_fill(pad_mask, 0)
            seq_emb = proj_seq.sum(dim=1) / mask.sum(dim=1)
        elif self.pool_type == 'final':
            idxs = mask.expand_as(proj_seq).sum(dim=1, keepdim=True).long() - 1
            seq_emb = proj_seq.gather(dim=1, index=idxs)
        return seq_emb

    @classmethod
    def from_params(cls, d_inp, d_proj, project=True):
        return cls(d_inp, d_proj=d_proj, project=project)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_inp': 4}]
