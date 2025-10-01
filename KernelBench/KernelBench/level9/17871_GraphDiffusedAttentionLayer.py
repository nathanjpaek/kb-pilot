import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphDiffusedAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphDiffusedAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features),
            dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a_1 = nn.Parameter(torch.zeros(size=(out_features, 1), dtype=
            torch.float))
        nn.init.xavier_uniform_(self.a_1.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(out_features, 1), dtype=
            torch.float))
        nn.init.xavier_uniform_(self.a_2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        logit_1 = torch.matmul(h, self.a_1)
        logit_2 = torch.matmul(h, self.a_2)
        logits = logit_1 + logit_2.permute(1, 0)
        e = self.leakyrelu(logits)
        zero_vec = -9000000000000000.0 * e.new_tensor([1.0])
        e = torch.where(adj > 0, e, zero_vec)
        mean_h = torch.mean(h, dim=0, keepdim=True)
        h_all = torch.cat([h, mean_h], 0)
        glob_logit_2 = torch.mm(mean_h, self.a_2)
        glob_logit = logit_1 + glob_logit_2
        e_diffused = self.leakyrelu(glob_logit)
        e_all = torch.cat([e, e_diffused], -1)
        attention = F.softmax(e_all, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_out = torch.mm(attention, h_all)
        return F.elu(h_out)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4, 'dropout': 0.5,
        'alpha': 4}]
