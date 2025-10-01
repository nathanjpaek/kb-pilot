import torch
import typing
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_sizes: 'typing.Iterable[int]',
        out_dim, activation_function=nn.Sigmoid(), activation_out=None):
        super(MLP, self).__init__()
        i_h_sizes = [input_dim] + hidden_sizes
        self.mlp = nn.Sequential()
        for idx in range(len(i_h_sizes) - 1):
            self.mlp.add_module('layer_{}'.format(idx), nn.Linear(
                in_features=i_h_sizes[idx], out_features=i_h_sizes[idx + 1]))
            self.mlp.add_module('act', activation_function)
        self.mlp.add_module('out_layer', nn.Linear(i_h_sizes[-1], out_dim))
        if activation_out is not None:
            self.mlp.add_module('out_layer_activation', activation_out)

    def init(self):
        for i, l in enumerate(self.mlp):
            if type(l) == nn.Linear:
                nn.init.xavier_normal_(l.weight)

    def forward(self, x):
        return self.mlp(x)


class GINPreTransition(nn.Module):

    def __init__(self, node_state_dim: 'int', node_label_dim: 'int',
        mlp_hidden_dim: 'typing.Iterable[int]', activation_function=nn.Tanh()):
        super(type(self), self).__init__()
        d_i = node_state_dim + node_label_dim
        d_o = node_state_dim
        d_h = list(mlp_hidden_dim)
        self.mlp = MLP(input_dim=d_i, hidden_sizes=d_h, out_dim=d_o,
            activation_function=activation_function, activation_out=
            activation_function)

    def forward(self, node_states, node_labels, edges, agg_matrix):
        intermediate_states = self.mlp(torch.cat([node_states, node_labels],
            -1))
        new_state = torch.matmul(agg_matrix, intermediate_states[edges[:, 0]]
            ) + torch.matmul(agg_matrix, intermediate_states[edges[:, 1]])
        return new_state


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.ones([4, 4],
        dtype=torch.int64), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'node_state_dim': 4, 'node_label_dim': 4, 'mlp_hidden_dim':
        [4, 4]}]
