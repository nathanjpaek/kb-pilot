import torch
import torch.optim
import torch.jit
import torch.nn as nn


class SimpleMLPGen_with_meta_feature(nn.Module):

    def __init__(self, num_in_features, num_out_features, neurons_per_layer):
        super(SimpleMLPGen_with_meta_feature, self).__init__()
        self.l_in = nn.Linear(in_features=num_in_features, out_features=
            neurons_per_layer)
        self.l_out = nn.Linear(in_features=neurons_per_layer, out_features=
            num_out_features)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.act(self.l_in(x))
        x = self.l_out(x)
        return x

    def set_parameters(self, meta_in_features, simple_mlp_gen_obj):
        x = simple_mlp_gen_obj.act(simple_mlp_gen_obj.l_in(meta_in_features))
        x = simple_mlp_gen_obj.l_out(x)
        _base = (simple_mlp_gen_obj.num_in_features * simple_mlp_gen_obj.
            neurons_per_layer)
        l_in_weight = x[:_base].reshape((simple_mlp_gen_obj.num_in_features,
            simple_mlp_gen_obj.neurons_per_layer)).t()
        l_in_bias = x[_base:_base + simple_mlp_gen_obj.neurons_per_layer]
        _base += simple_mlp_gen_obj.neurons_per_layer
        _base_add = (simple_mlp_gen_obj.neurons_per_layer *
            simple_mlp_gen_obj.num_out_features)
        l_out_weight = x[_base:_base + _base_add].reshape((
            simple_mlp_gen_obj.neurons_per_layer, simple_mlp_gen_obj.
            num_out_features)).t()
        _base += _base_add
        l_out_bias = x[_base:]
        self.l_in.weight = torch.nn.Parameter(l_in_weight)
        self.l_out.weight = torch.nn.Parameter(l_out_weight)
        self.l_in.bias = torch.nn.Parameter(l_in_bias)
        self.l_out.bias = torch.nn.Parameter(l_out_bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_in_features': 4, 'num_out_features': 4,
        'neurons_per_layer': 1}]
