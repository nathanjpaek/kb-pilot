import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import init
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, act=nn.ReLU(),
        normalize_input=True):
        super(MLP, self).__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, output_dim)
        self.act = act
        self.normalize_input = normalize_input
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn
                    .init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        if self.normalize_input:
            x = (x - torch.mean(x, dim=0)) / torch.std(x, dim=0)
        x = self.act(self.linear_1(x))
        return self.linear_2(x)


class GraphConv(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim,
        normalize_embedding=False, normalize_embedding_l2=False, att=False,
        mpnn=False, graphsage=False):
        super(GraphConv, self).__init__()
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_l2 = normalize_embedding_l2
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.att = att
        self.mpnn = mpnn
        self.graphsage = graphsage
        if self.graphsage:
            self.out_compute = MLP(input_dim=input_dim * 2, hidden_dim=
                hidden_dim, output_dim=output_dim, act=nn.ReLU(),
                normalize_input=False)
        elif self.mpnn:
            self.out_compute = MLP(input_dim=hidden_dim, hidden_dim=
                hidden_dim, output_dim=output_dim, act=nn.ReLU(),
                normalize_input=False)
        else:
            self.out_compute = MLP(input_dim=input_dim, hidden_dim=
                hidden_dim, output_dim=output_dim, act=nn.ReLU(),
                normalize_input=False)
        if self.att:
            self.att_compute = MLP(input_dim=input_dim, hidden_dim=
                hidden_dim, output_dim=output_dim, act=nn.LeakyReLU(0.2),
                normalize_input=False)
        if self.mpnn:
            self.mpnn_compute = MLP(input_dim=input_dim * 2, hidden_dim=
                hidden_dim, output_dim=hidden_dim, act=nn.ReLU(),
                normalize_input=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, adj):
        if self.att:
            x_att = self.att_compute(x)
            att = x_att @ x_att.permute(1, 0)
            att = self.softmax(att)
            pred = torch.matmul(adj * att, x)
        elif self.mpnn:
            x1 = x.unsqueeze(0).repeat(x.shape[0], 1, 1)
            x2 = x.unsqueeze(1).repeat(1, x.shape[0], 1)
            e = torch.cat((x1, x2), dim=-1)
            e = self.mpnn_compute(e)
            pred = torch.mean(adj.unsqueeze(-1) * e, dim=1)
        else:
            pred = torch.matmul(adj, x)
        if self.graphsage:
            pred = torch.cat((pred, x), dim=-1)
        pred = self.out_compute(pred)
        if self.normalize_embedding:
            pred = (pred - torch.mean(pred, dim=0)) / torch.std(pred, dim=0)
        if self.normalize_embedding_l2:
            pred = F.normalize(pred, p=2, dim=-1)
        return pred


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'hidden_dim': 4}]
