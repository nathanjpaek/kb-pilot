import torch
from torch import Tensor
import torch.distributed as dist
import torch.nn as nn
from torch.nn import Linear


class ParallelModule(nn.Module):
    """Parents of all parallel layer classes"""

    def __init__(self):
        super().__init__()
        self.mp_group = None

    def allreduce(self, outputs):
        if self.mp_group is not None and dist.get_world_size(group=self.
            mp_group) > 1:
            dist.all_reduce(outputs, group=self.mp_group)
        return outputs


class AllReduceLinear(Linear, ParallelModule):
    """All-reduce linear layer"""

    def forward(self, input: 'Tensor') ->Tensor:
        outputs = input.matmul(self.weight.t())
        self.allreduce(outputs)
        if self.bias is not None:
            outputs += self.bias
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
