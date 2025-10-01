from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class Normalization(nn.Module):

    def __init__(self, cfg):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(cfg.embedding_dim,
            elementwise_affine=True)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.
            size())


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cfg': _mock_config(embedding_dim=4)}]
