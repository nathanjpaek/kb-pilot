from torch.nn import Module
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GATLayer(Module):

    def __init__(self, input_channel, output_channel, use_bias=True):
        super(GATLayer, self).__init__()
        self.use_bias = use_bias
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel = Parameter(torch.FloatTensor(input_channel,
            output_channel))
        torch.nn.init.xavier_uniform_(self.kernel, gain=1.414)
        if self.use_bias:
            self.bias = Parameter(torch.FloatTensor(output_channel))
            torch.nn.init.constant_(self.bias, 0)
        self.attn_kernel_self = Parameter(torch.FloatTensor(output_channel, 1))
        torch.nn.init.xavier_uniform_(self.attn_kernel_self, gain=1.414)
        self.attn_kernel_neighs = Parameter(torch.FloatTensor(
            output_channel, 1))
        torch.nn.init.xavier_uniform_(self.attn_kernel_neighs, gain=1.414)

    def forward(self, X, adj):
        features = torch.mm(X, self.kernel)
        attn_for_self = torch.mm(features, self.attn_kernel_self)
        attn_for_neigh = torch.mm(features, self.attn_kernel_neighs)
        attn = attn_for_self + attn_for_neigh.t()
        attn = F.leaky_relu(attn, negative_slope=0.2)
        mask = -10000000000.0 * torch.ones_like(attn)
        attn = torch.where(adj > 0, attn, mask)
        attn = torch.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=0.5, training=self.training)
        features = F.dropout(features, p=0.5, training=self.training)
        node_features = torch.mm(attn, features)
        if self.use_bias:
            node_features = node_features + self.bias
        return node_features


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_channel': 4, 'output_channel': 4}]
