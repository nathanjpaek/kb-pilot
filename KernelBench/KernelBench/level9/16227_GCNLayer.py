import torch
import torch.nn as nn


class GCNLayer(nn.Module):

    def __init__(self, input_dim, output_dim, prop_depth=1, act=torch.relu,
        dropout=0.0, layer_i=0):
        super(GCNLayer, self).__init__()
        self.prop_depth = 1
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim, dtype
            =torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.weight.data)
        self.act = act
        self.dropout = nn.Dropout(p=dropout)
        self.layer_i = layer_i
        self.last_layer_flag = False

    def layer(self, x, adj_batch):
        x = x.transpose(0, 1)
        adj_batch = adj_batch[:, 1, :, :]
        x = torch.matmul(x, self.weight)
        x = torch.matmul(adj_batch, x)
        x = x.transpose(0, 1)
        return x

    def forward(self, x, adj_batch):
        x = self.layer(x, adj_batch)
        x = self.act(x)
        x = self.dropout(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
