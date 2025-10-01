from _paritybench_helpers import _mock_config
from torch.nn import Module
import math
import torch
import torch.nn as nn
import torch.nn.functional as f


class Module(nn.Module):

    def __init__(self):
        super().__init__()
        self.summary = {}

    def add_summary(self, name, val):
        if self.training:
            self.summary[name] = val.clone().detach().cpu().numpy()

    def get_summary(self, base_name=''):
        summary = {}
        if base_name:
            base_name += '/'
        if self.summary:
            summary.update({(base_name + name): val for name, val in self.
                summary.items()})
        for name, child in self.named_children():
            if hasattr(child, 'get_summary'):
                name = base_name + name
                summary.update(child.get_summary(name))
        return summary


class Alignment(Module):

    def __init__(self, args, __):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(args.
            hidden_size)))

    def _attention(self, a, b):
        return torch.matmul(a, b.transpose(1, 2)) * self.temperature

    def forward(self, a, b, mask_a, mask_b):
        attn = self._attention(a, b)
        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float()
            ).bool()
        attn.masked_fill_(~mask, -10000000.0)
        attn_a = f.softmax(attn, dim=1)
        attn_b = f.softmax(attn, dim=2)
        feature_b = torch.matmul(attn_a.transpose(1, 2), a)
        feature_a = torch.matmul(attn_b, b)
        self.add_summary('temperature', self.temperature)
        self.add_summary('attention_a', attn_a)
        self.add_summary('attention_b', attn_b)
        return feature_a, feature_b


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(hidden_size=4), '__': 4}]
