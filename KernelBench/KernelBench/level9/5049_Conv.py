import torch
from torch import nn
import torch.utils.data
import torch.nn.init as init


def initial_parameter(net, initial_method=None):
    """A method used to initialize the weights of PyTorch models.

    :param net: a PyTorch model
    :param initial_method: str, one of the following initializations

            - xavier_uniform
            - xavier_normal (default)
            - kaiming_normal, or msra
            - kaiming_uniform
            - orthogonal
            - sparse
            - normal
            - uniform

    """
    if initial_method == 'xavier_uniform':
        init_method = init.xavier_uniform_
    elif initial_method == 'xavier_normal':
        init_method = init.xavier_normal_
    elif initial_method == 'kaiming_normal' or initial_method == 'msra':
        init_method = init.kaiming_normal_
    elif initial_method == 'kaiming_uniform':
        init_method = init.kaiming_uniform_
    elif initial_method == 'orthogonal':
        init_method = init.orthogonal_
    elif initial_method == 'sparse':
        init_method = init.sparse_
    elif initial_method == 'normal':
        init_method = init.normal_
    elif initial_method == 'uniform':
        init_method = init.uniform_
    else:
        init_method = init.xavier_normal_

    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m
            , nn.Conv3d):
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for w in m.parameters():
                if len(w.data.size()) > 1:
                    init_method(w.data)
                else:
                    init.normal_(w.data)
        elif hasattr(m, 'weight') and m.weight.requires_grad:
            init_method(m.weight.data)
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)
                    else:
                        init.normal_(w.data)
    net.apply(weights_init)


class Conv(nn.Module):
    """
    Basic 1-d convolution module.
    initialize with xavier_uniform
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, activation='relu',
        initial_method=None):
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, dilation=dilation, groups=groups, bias=bias)
        activations = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}
        if activation in activations:
            self.activation = activations[activation]
        else:
            raise Exception('Should choose activation function from: ' +
                ', '.join([x for x in activations]))
        initial_parameter(self, initial_method)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        x = self.activation(x)
        x = torch.transpose(x, 1, 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
