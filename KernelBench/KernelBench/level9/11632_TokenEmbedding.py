import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.quantization
import torch.onnx
import torch.nn.parallel
import torch.utils.data
import torch.fx
import torch.nn
import torch.optim
import torch.profiler


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size: 'int', emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: 'Tensor'):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'vocab_size': 4, 'emb_size': 4}]
