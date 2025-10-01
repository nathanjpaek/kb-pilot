import torch
import torch.nn as nn
import torch.utils.data
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from abc import ABC
from abc import abstractmethod


class BaseMiner(nn.Module, ABC):

    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        super().__init__()

    @abstractmethod
    def forward(self, anchor: 'torch.Tensor', target: 'torch.Tensor') ->Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class UniformBatchMiner(BaseMiner):

    def __init__(self, sample_size: 'int'):
        super().__init__()
        self.sample_size = sample_size

    def forward(self, anchor: 'torch.Tensor', target: 'torch.Tensor') ->Tuple[
        torch.Tensor, torch.Tensor]:
        batch_size = target.size(0)
        rand_idx = torch.randint(0, batch_size, (self.sample_size *
            batch_size,))
        neg_samples = target[rand_idx].unsqueeze(1)
        pos_samples = target.unsqueeze(1)
        anchor = anchor.unsqueeze(1)
        repeated = torch.repeat_interleave(anchor, self.sample_size, dim=0)
        pos = torch.cat([anchor, pos_samples], dim=1)
        neg = torch.cat([repeated, neg_samples], dim=1)
        return pos, neg


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'sample_size': 4}]
