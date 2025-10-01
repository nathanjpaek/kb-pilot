import torch
import torch.nn as nn


def standardize(param, assert_length):
    if type(param) is not list and type(param) is not tuple:
        param = [param] * assert_length
    assert len(param
        ) == assert_length, 'expect %s input params, got %s input parameter' % (
        assert_length, len(param))
    return param


def fc_layer(input, layer_size, bias=True, name=None, activation=nn.Sigmoid
    (), batch_norm=None, dropout=0):
    layer_size = [input] + [layer_size] if type(layer_size) is not list else [
        input] + layer_size
    assert_length = len(layer_size) - 1
    bias = standardize(bias, assert_length)
    activation = standardize(activation, assert_length)
    batch_norm = standardize(batch_norm, assert_length)
    dropout = standardize(dropout, assert_length)
    if name is None:
        name = ''
    modules = nn.Sequential()
    for i in range(len(layer_size) - 1):
        modules.add_module(name + '_fc_' + str(i), nn.Linear(layer_size[i],
            layer_size[i + 1], bias[i]))
        if batch_norm[i]:
            modules.add_module(name + 'bn_' + str(i), batch_norm[i](
                layer_size[i + 1]))
        if activation[i]:
            modules.add_module(name + 'act_' + str(i), activation[i])
        if dropout[i] > 0:
            modules.add_module(name + 'drop_' + str(i), nn.Dropout2d(
                dropout[i]))
    return modules


class FC_Layer(nn.Module):

    def __init__(self, input, layer_size, bias=True, name=None, activation=
        nn.Sigmoid(), batch_norm=None, dropout=0):
        super().__init__()
        self.fc_layer = fc_layer(input, layer_size, bias=bias, name=name,
            activation=activation, batch_norm=batch_norm, dropout=dropout)

    def forward(self, x, batch_dim=0):
        if len(x.shape):
            x = x.view(x.size(batch_dim), -1)
        return self.fc_layer.forward(x)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input': 4, 'layer_size': 1}]
