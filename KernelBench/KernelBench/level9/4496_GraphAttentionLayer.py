import torch
import torch.nn as nn
import torch.nn.functional as F


def module_test_print(var_input, var_inmed, var_ouput):
    for var in (var_input, var_inmed, var_ouput):
        None
        for key, value in var.items():
            None
            None


class GraphAttentionLayer(nn.Module):

    def __init__(self, dim_input, dim_output, dropout=0.0,
        negative_slope_LeakyRelu=0.01, module_test=False):
        super(GraphAttentionLayer, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dropout = dropout
        self.W = nn.Parameter(torch.empty(size=(dim_input, dim_output)))
        nn.init.xavier_uniform_(self.W.data, gain=nn.init.calculate_gain(
            'leaky_relu', negative_slope_LeakyRelu))
        self.leakyrelu = nn.LeakyReLU(negative_slope_LeakyRelu)
        self.module_test = module_test

    def _attention_score(self, x, y=None):
        return torch.matmul(x, x.transpose(-2, -1))

    def forward(self, x, adj, A):
        """
        input:
            x   : (batch_size, n_node, dim_node)
            adj : (batch_size, n_node, n_node)
        """
        xW = torch.matmul(x, self.W)
        score = self._attention_score(xW)
        score += A
        zero_vec = -9000000000000000.0 * torch.ones_like(score)
        _attention = torch.where(adj > 0, score, zero_vec)
        attention = F.softmax(_attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        x_new = F.elu(torch.matmul(attention, xW))
        if self.module_test:
            var_input = ['x', 'adj']
            var_inmed = ['xW', 'score', 'zero_vec', '_attention', 'attention']
            var_ouput = ['x_new']
            locals_cap = locals()
            module_test_print(dict(zip(var_input, [eval(v, locals_cap) for
                v in var_input])), dict(zip(var_inmed, [eval(v, locals_cap) for
                v in var_inmed])), dict(zip(var_ouput, [eval(v, locals_cap) for
                v in var_ouput])))
        return x_new

    def __repr__(self):
        return '{}({}->{})'.format(self.__class__.__name__, self.dim_input,
            self.dim_output)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_input': 4, 'dim_output': 4}]
