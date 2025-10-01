import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.onnx
import torch.optim
import torch.utils.data.distributed


class MLPTanH(nn.Module):

    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(MLPTanH, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential()
        linear = nn.Linear(input_dim, hidden_dim)
        self.layers.add_module('fc_0', linear)
        self.layers.add_module('Tanh_0', nn.Tanh())
        self.layers.add_module('fc_1', nn.Linear(hidden_dim, vocab_size))

    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))

    def set_decoder_weights(self, embedding_weights):
        self.layers.fc_1.weight = embedding_weights


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4, 'vocab_size': 4}]
