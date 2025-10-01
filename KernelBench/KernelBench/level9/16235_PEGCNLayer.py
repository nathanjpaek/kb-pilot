import torch
import torch.nn as nn


class PEGCNLayer(nn.Module):

    def __init__(self, input_dim, output_dim, prop_depth, act=torch.relu,
        dropout=0.0, layer_i=0):
        super(PEGCNLayer, self).__init__()
        self.prop_depth = prop_depth
        self.act = act
        self.weight = nn.Parameter(torch.empty(1, prop_depth, input_dim,
            output_dim, dtype=torch.float), requires_grad=True)
        nn.init.uniform_(self.weight.data)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.layer_i = layer_i
        self.last_layer_flag = False

    def layer(self, x, adj_batch):
        if adj_batch.dim() < 4:
            adj_batch = adj_batch.unsqueeze(0)
        x = x.transpose(0, 1).unsqueeze(dim=1).repeat(1, self.prop_depth, 1, 1)
        x = torch.matmul(x, self.weight)
        x = torch.matmul(adj_batch, x)
        x = x.sum(dim=1)
        x = x.transpose(0, 1)
        return x

    def forward(self, x, adj_batch):
        x = self.layer(x, adj_batch)
        x = self.act(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'prop_depth': 1}]
