import torch
import torch.cuda
from torch import nn
import torch.nn
import torch.utils.data
import torch.fx
import torch.utils.tensorboard._pytorch_graph


class TwoLinearsModel(nn.Module):

    def __init__(self, per_sample_shape: 'list', hidden_size: 'int',
        output_size: 'int'):
        super(TwoLinearsModel, self).__init__()
        assert len(per_sample_shape) == 3
        self.per_sample_shape = per_sample_shape
        input_size = per_sample_shape[0]
        for dim in per_sample_shape[1:]:
            input_size *= dim
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: 'torch.Tensor'):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'per_sample_shape': [4, 4, 4], 'hidden_size': 4,
        'output_size': 4}]
