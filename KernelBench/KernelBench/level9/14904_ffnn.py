import torch
import torch.nn as nn
import torch.utils.data.dataloader
import torch.nn


def get_shape(t):
    return list(t.shape)


class ffnn(nn.Module):

    def __init__(self, emb_size, num_layers, hidden_size, output_size,
        dropout, output_weights_initializer=None):
        super(ffnn, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.weights = nn.Parameter(torch.Tensor(emb_size, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        self.activation = torch.nn.ReLU()
        self.num_layers = num_layers
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.initializer = output_weights_initializer
        self.initialize()

    def initialize(self):
        if self.initializer is None:
            torch.nn.init.xavier_uniform_(self.weights, gain=1)
        else:
            self.initializer(self.weights, gain=1)
        nn.init.zeros_(self.bias)

    def forward(self, inputs):
        current_inputs = inputs
        if len(get_shape(inputs)) == 3:
            batch_size, seqlen, emb_size = get_shape(inputs)
            current_inputs = inputs.reshape(batch_size * seqlen, emb_size)
        emb_size = get_shape(current_inputs)[-1]
        assert emb_size == self.emb_size, 'last dim of input does not match this layer'
        outputs = current_inputs.matmul(self.weights) + self.bias
        if len(get_shape(inputs)) == 3:
            outputs = outputs.reshape(batch_size, seqlen, self.output_size)
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'emb_size': 4, 'num_layers': 1, 'hidden_size': 4,
        'output_size': 4, 'dropout': 0.5}]
