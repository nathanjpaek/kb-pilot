import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


def identity(x: 'torch.Tensor') ->torch.Tensor:
    """Return input without any change."""
    return x


def init_layer_uniform(layer: 'nn.Linear', init_w: 'float'=0.003) ->nn.Linear:
    """Init uniform parameters on the single layer"""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer


class MLP(nn.Module):
    """Baseline of Multilayer perceptron.

    Attributes:
        input_size (int): size of input
        output_size (int): size of output layer
        hidden_sizes (list): sizes of hidden layers
        hidden_activation (function): activation function of hidden layers
        output_activation (function): activation function of output layer
        hidden_layers (list): list containing linear layers
        use_output_layer (bool): whether or not to use the last layer
        n_category (int): category number (-1 if the action is continuous)

    """

    def __init__(self, input_size: 'int', output_size: 'int', hidden_sizes:
        'list', hidden_activation: 'Callable'=F.relu, output_activation:
        'Callable'=identity, linear_layer: 'nn.Module'=nn.Linear,
        use_output_layer: 'bool'=True, n_category: 'int'=-1, init_fn:
        'Callable'=init_layer_uniform):
        """Initialize.

        Args:
            input_size (int): size of input
            output_size (int): size of output layer
            hidden_sizes (list): number of hidden layers
            hidden_activation (function): activation function of hidden layers
            output_activation (function): activation function of output layer
            linear_layer (nn.Module): linear layer of mlp
            use_output_layer (bool): whether or not to use the last layer
            n_category (int): category number (-1 if the action is continuous)
            init_fn (Callable): weight initialization function bound for the last layer

        """
        super(MLP, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.linear_layer = linear_layer
        self.use_output_layer = use_output_layer
        self.n_category = n_category
        self.hidden_layers: 'list' = []
        in_size = self.input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = self.linear_layer(in_size, next_size)
            in_size = next_size
            self.__setattr__('hidden_fc{}'.format(i), fc)
            self.hidden_layers.append(fc)
        if self.use_output_layer:
            self.output_layer = self.linear_layer(in_size, output_size)
            self.output_layer = init_fn(self.output_layer)
        else:
            self.output_layer = identity
            self.output_activation = identity

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward method implementation."""
        for hidden_layer in self.hidden_layers:
            x = self.hidden_activation(hidden_layer(x))
        x = self.output_activation(self.output_layer(x))
        return x


class NoisyMLPHandler:
    """Includes methods to handle noisy linear."""

    def reset_noise(self):
        """Re-sample noise"""
        for _, module in self.named_children():
            module.reset_noise()


class DuelingMLP(MLP, NoisyMLPHandler):
    """Multilayer perceptron with dueling construction."""

    def __init__(self, input_size: 'int', output_size: 'int', hidden_sizes:
        'list', hidden_activation: 'Callable'=F.relu, linear_layer:
        'nn.Module'=nn.Linear, init_fn: 'Callable'=init_layer_uniform):
        """Initialize."""
        super(DuelingMLP, self).__init__(input_size=input_size, output_size
            =output_size, hidden_sizes=hidden_sizes, hidden_activation=
            hidden_activation, linear_layer=linear_layer, use_output_layer=
            False)
        in_size = hidden_sizes[-1]
        self.advantage_hidden_layer = self.linear_layer(in_size, in_size)
        self.advantage_layer = self.linear_layer(in_size, output_size)
        self.advantage_layer = init_fn(self.advantage_layer)
        self.value_hidden_layer = self.linear_layer(in_size, in_size)
        self.value_layer = self.linear_layer(in_size, 1)
        self.value_layer = init_fn(self.value_layer)

    def _forward_dueling(self, x: 'torch.Tensor') ->torch.Tensor:
        adv_x = self.hidden_activation(self.advantage_hidden_layer(x))
        val_x = self.hidden_activation(self.value_hidden_layer(x))
        advantage = self.advantage_layer(adv_x)
        value = self.value_layer(val_x)
        advantage_mean = advantage.mean(dim=-1, keepdim=True)
        q = value + advantage - advantage_mean
        return q

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward method implementation."""
        x = super(DuelingMLP, self).forward(x)
        x = self._forward_dueling(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4, 'hidden_sizes': [4, 4]}]
