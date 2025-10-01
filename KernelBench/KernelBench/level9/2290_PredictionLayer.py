import torch
import torch.nn as nn
import torch.utils.data


class PredictionLayer(nn.Module):

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ['binary', 'multiclass', 'regression']:
            raise ValueError('task must be binary, multiclass or regression')
        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, X):
        output = X
        if self.use_bias:
            output += self.bias
        if self.task == 'binary':
            output = torch.sigmoid(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
