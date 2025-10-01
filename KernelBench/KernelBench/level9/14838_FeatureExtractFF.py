import torch
import torch.utils.data
import torch.nn as nn


class FeatureExtractFF(nn.Module):

    def __init__(self, input_dim, hidden_sizes=(15,), activation_fn=nn.ReLU,
        **activation_args):
        super(FeatureExtractFF, self).__init__()
        self._in = input_dim
        self._hidden_sizes = hidden_sizes
        self._activation_fn = activation_fn
        self._activation_args = activation_args
        self.feature = nn.Sequential()
        hin = self._in
        for i, h in enumerate(self._hidden_sizes):
            self.feature.add_module(f'f_fc{i}', nn.Linear(hin, h))
            self.feature.add_module(f'f_{activation_fn.__name__}{i}',
                activation_fn(**activation_args))
            hin = h
        self._out_features = hin

    def forward(self, input_data):
        return self.feature(input_data)

    def extra_repr(self):
        return f'FC: {self.hidden_sizes}x{self._activation_fn.__name__}'

    def hidden_layer(self, index=0):
        return self.feature[index * 2]

    def output_size(self):
        return self._out_features


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
