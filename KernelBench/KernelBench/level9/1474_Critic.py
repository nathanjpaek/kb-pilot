import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size,
        hidden_layers=None, action_input_layer=0, init_type='normal',
        activation='leaky_relu', init_std=0.01):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation
        self.layers = nn.ModuleList()
        input_size = self.state_size + action_size + action_parameter_size
        last_hidden_layer_size = input_size
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(input_size, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1],
                    hidden_layers[i]))
            last_hidden_layer_size = hidden_layers[nh - 1]
        self.output_layer = nn.Linear(last_hidden_layer_size, 1)
        for i in range(0, len(self.layers)):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(self.layers[i].weight.data,
                    nonlinearity=self.activation)
            elif init_type == 'normal':
                nn.init.normal_(self.layers[i].weight.data, std=init_std)
            else:
                raise ValueError('Unknown init_type ' + str(init_type))
            nn.init.zeros_(self.layers[i].bias.data)
        nn.init.normal_(self.output_layer.weight, std=init_std)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, state, actions, action_parameters):
        x = torch.cat((state, actions, action_parameters), dim=1)
        negative_slope = 0.01
        num_hidden_layers = len(self.layers)
        for i in range(0, num_hidden_layers):
            if self.activation == 'relu':
                x = F.relu(self.layers[i](x))
            elif self.activation == 'leaky_relu':
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError('Unknown activation function ' + str(self.
                    activation))
        Q = self.output_layer(x)
        return Q


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'action_parameter_size': 4}
        ]
