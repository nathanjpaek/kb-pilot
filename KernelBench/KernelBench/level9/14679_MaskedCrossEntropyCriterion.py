import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss


class MaskedCrossEntropyCriterion(_WeightedLoss):

    def __init__(self, ignore_index=[-100], reduce=None):
        super(MaskedCrossEntropyCriterion, self).__init__()
        self.padding_idx = ignore_index
        self.reduce = reduce

    def forward(self, outputs, targets):
        lprobs = nn.functional.log_softmax(outputs, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        for idx in self.padding_idx:
            targets[targets == idx] = 0
        nll_loss = -lprobs.gather(dim=-1, index=targets.unsqueeze(1))
        if self.reduce:
            nll_loss = nll_loss.sum()
        return nll_loss.squeeze()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {}]
