import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size,
        hidden_layers=None, init_std=0.01, init_type='normal', activation=
        'leaky_relu', squashing_function=False):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.squashing_function = squashing_function
        assert self.squashing_function is False
        self.activation = activation
        self.layers = nn.ModuleList()
        last_hidden_layer_size = self.state_size
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(self.state_size, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1],
                    hidden_layers[i]))
            last_hidden_layer_size = hidden_layers[nh - 1]
        self.action_output_layer = nn.Linear(last_hidden_layer_size, self.
            action_size)
        self.action_parameters_output_layer = nn.Linear(last_hidden_layer_size,
            self.action_parameter_size)
        for i in range(0, len(self.layers)):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(self.layers[i].weight.data,
                    nonlinearity=self.activation)
            elif init_type == 'normal':
                nn.init.normal_(self.layers[i].weight.data, std=init_std)
            else:
                raise ValueError('Unknown init_type ' + str(init_type))
            nn.init.zeros_(self.layers[i].bias.data)
        nn.init.normal_(self.action_output_layer.weight, std=init_std)
        nn.init.zeros_(self.action_output_layer.bias)
        nn.init.normal_(self.action_parameters_output_layer.weight, std=
            init_std)
        nn.init.zeros_(self.action_parameters_output_layer.bias)
        self.action_parameters_passthrough_layer = nn.Linear(self.
            state_size, self.action_parameter_size)
        nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        nn.init.zeros_(self.action_parameters_passthrough_layer.bias)
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        negative_slope = 0.01
        x = state
        num_hidden_layers = len(self.layers)
        for i in range(0, num_hidden_layers):
            if self.activation == 'relu':
                x = F.relu(self.layers[i](x))
            elif self.activation == 'leaky_relu':
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError('Unknown activation function ' + str(self.
                    activation))
        actions = self.action_output_layer(x)
        action_params = self.action_parameters_output_layer(x)
        action_params += self.action_parameters_passthrough_layer(state)
        return actions, action_params


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'action_parameter_size': 4}
        ]
