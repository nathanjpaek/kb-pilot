import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT(nn.Module):

    def __init__(self, num_feats):
        super(GAT, self).__init__()
        self.num_feats = num_feats
        self.weight_key = nn.Parameter(torch.zeros(size=(self.num_feats, 1)))
        self.weight_query = nn.Parameter(torch.zeros(size=(self.num_feats, 1)))
        nn.init.xavier_uniform_(self.weight_key, gain=1.414)
        nn.init.xavier_uniform_(self.weight_query, gain=1.414)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        :param x: dim: bz x num_node x num_feat
        :return: dim: bz x num_node x num_node
        """
        batch_size = x.size(0)
        num_nodes = x.size(1)
        key = torch.matmul(x, self.weight_key)
        query = torch.matmul(x, self.weight_query)
        attn_input = key.repeat(1, 1, num_nodes).view(batch_size, num_nodes *
            num_nodes, 1) + query.repeat(1, num_nodes, 1)
        attn_output = attn_input.squeeze(2).view(batch_size, num_nodes,
            num_nodes)
        attn_output = F.leaky_relu(attn_output, negative_slope=0.2)
        attention = F.softmax(attn_output, dim=2)
        attention = self.dropout(attention)
        attn_feat = torch.matmul(attention, x).permute(0, 2, 1)
        return attn_feat


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'num_feats': 4}]
