from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class ShallowCombination(nn.Module):
    """This Module can be used to generate a shallow combination from two embeddings using a gate."""

    def __init__(self, bertram_config: 'BertramConfig'):
        super(ShallowCombination, self).__init__()
        self.linear = nn.Linear(2 * bertram_config.output_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.mode = bertram_config.mode

    def forward(self, embs1, embs2):
        embs_combined = torch.cat([embs1, embs2], dim=-1)
        a = self.sigmoid(self.linear(embs_combined))
        return a * embs1 + (1 - a) * embs2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'bertram_config': _mock_config(output_size=4, mode=4)}]
