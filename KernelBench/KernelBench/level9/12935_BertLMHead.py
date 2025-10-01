from _paritybench_helpers import _mock_config
from torch.nn import Module
import torch
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn.functional import gelu


class BertLMHead(Module):

    def __init__(self, config):
        super(BertLMHead, self).__init__()
        hidden_size = config['hidden_size']
        self.mlp = Linear(hidden_size, hidden_size, bias=True)
        self.unembedding = Linear(hidden_size, config['vocab_size'], bias=True)
        self.layer_norm = LayerNorm((hidden_size,))

    def forward(self, activations):
        return self.unembedding(self.layer_norm(gelu(self.mlp(activations))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, vocab_size=4)}]
