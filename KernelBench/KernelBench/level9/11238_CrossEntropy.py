import torch
import torch.nn as nn


class CrossEntropy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, props, tgt):
        tgt_props = props.gather(2, tgt.unsqueeze(2)).squeeze()
        mask = (tgt > 0).float()
        return -(tgt_props * mask).sum() / mask.sum()


def get_inputs():
    return [torch.ones([4, 4, 4], dtype=torch.int64), torch.ones([4, 4],
        dtype=torch.int64)]


def get_init_inputs():
    return [[], {}]
