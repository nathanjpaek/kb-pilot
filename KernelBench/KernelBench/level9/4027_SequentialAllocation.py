from torch.nn import Module
import torch
from torch.nn import functional as F
from torch.nn import Linear


def _sequential_allocation(p, weights):
    _, slots, bidders_plus_one = p.shape
    bidders = bidders_plus_one - 1
    cumulative_total = p[:, 0, :bidders]
    if weights is None:
        alloc = cumulative_total
    else:
        alloc = cumulative_total * weights[0]
    for k in range(1, slots):
        slot_total = (1 - cumulative_total) * p[:, k, :bidders] * (1 - p[:,
            k - 1, [bidders for _ in range(bidders)]])
        if weights is None:
            alloc = alloc + slot_total
        else:
            alloc = alloc + slot_total * weights[k]
        cumulative_total = cumulative_total + slot_total
    return alloc


class SequentialAllocation(Module):
    __constants__ = ['in_features', 'bidders', 'slots', 'weights']

    def __init__(self, in_features, slots, bidders, weights=None):
        super(SequentialAllocation, self).__init__()
        self.in_features = in_features
        self.slots = slots
        self.bidders = bidders
        self.weights = weights
        self.linear = Linear(in_features, slots * (bidders + 1))

    def forward(self, x):
        probs = F.softmax(self.linear(x).reshape(-1, self.slots, self.
            bidders + 1), dim=2)
        return _sequential_allocation(probs, weights=self.weights)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'slots': 4, 'bidders': 4}]
