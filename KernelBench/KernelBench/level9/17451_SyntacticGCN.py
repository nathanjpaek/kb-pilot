import torch
import torch.nn as nn
import torch.nn.functional as F


class SyntacticGCN(nn.Module):

    def __init__(self, input_size, hidden_size, num_labels, bias=True):
        super(SyntacticGCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.W = nn.Parameter(torch.empty(num_labels, input_size,
            hidden_size, dtype=torch.float))
        nn.init.xavier_normal_(self.W)
        if bias:
            self.bias = True
            self.b = nn.Parameter(torch.empty(num_labels, hidden_size,
                dtype=torch.float))
            nn.init.xavier_normal_(self.b)

    def forward(self, graph, nodes):
        b, n, _ = nodes.size()
        l, input_size, hidden_size = (self.num_labels, self.input_size,
            self.hidden_size)
        g = graph.transpose(2, 3).float().contiguous().view(b, n * l, n)
        x = g.bmm(nodes).view(b, n, l * input_size)
        h = x.matmul(self.W.view(l * input_size, hidden_size))
        if self.bias:
            bias = (graph.float().view(b * n * n, l) @ self.b).view(b, n, n,
                hidden_size)
            bias = bias.sum(2)
            h = h + bias
        norm = graph.view(b, n, n * l).sum(-1).float().unsqueeze(-1) + 1e-10
        hidden = F.relu(h / norm)
        return hidden


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'num_labels': 4}]
