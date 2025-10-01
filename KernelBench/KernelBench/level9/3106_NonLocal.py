import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.jit
import torch.nn


class NonLocal(nn.Module):

    def __init__(self, in_size, attention_size=32, size=None, scale=None):
        super(NonLocal, self).__init__()
        self.size = size
        self.scale = scale
        self.attention_size = attention_size
        self.query = nn.Conv2d(in_size, attention_size, 1)
        self.key = nn.Conv2d(in_size, attention_size, 1)
        self.value = nn.Conv2d(in_size, attention_size, 1)
        self.project = nn.Conv2d(attention_size, in_size, 1)

    def forward(self, inputs):
        scaled_inputs = None
        if self.scale:
            scaled_inputs = func.max_pool2d(inputs, self.scale)
        elif self.size:
            scaled_inputs = func.adaptive_max_pool2d(inputs, self.size)
        else:
            scaled_inputs = inputs
        query = self.query(inputs).view(inputs.size(0), self.attention_size, -1
            )
        key = self.key(scaled_inputs).view(scaled_inputs.size(0), self.
            attention_size, -1)
        value = self.value(scaled_inputs).view(scaled_inputs.size(0), self.
            attention_size, -1)
        key = key.permute(0, 2, 1)
        assignment = (key @ query).softmax(dim=1)
        result = value @ assignment
        result = result.view(inputs.size(0), self.attention_size, *inputs.
            shape[2:])
        return self.project(result) + inputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4}]
