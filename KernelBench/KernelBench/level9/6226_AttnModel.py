import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """
        Multi-Layer Perceptron
        :param in_dim: int, size of input feature
        :param n_classes: int, number of output classes
        :param hidden_dim: int, size of hidden vector
        :param dropout: float, dropout rate
        :param n_layers: int, number of layers, at least 2, default = 2
        :param act: function, activation function, default = leaky_relu
    """

    def __init__(self, in_dim, n_classes, hidden_dim, dropout, n_layers=2,
        act=F.leaky_relu):
        super(MLP, self).__init__()
        self.l_in = nn.Linear(in_dim, hidden_dim)
        self.l_hs = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in
            range(n_layers - 2))
        self.l_out = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act
        return

    def forward(self, input):
        """
            :param input: Tensor of (batch_size, in_dim), input feature
            :returns: Tensor of (batch_size, n_classes), output class
        """
        hidden = self.act(self.l_in(self.dropout(input)))
        for l_h in self.l_hs:
            hidden = self.act(l_h(self.dropout(hidden)))
        output = self.l_out(self.dropout(hidden))
        return output


class AttnModel(nn.Module):
    """
        Attention Model
        :param dim: int, size of hidden vector
        :param dropout: float, dropout rate of attention model
    """

    def __init__(self, dim, dropout):
        super(AttnModel, self).__init__()
        self.score = MLP(dim * 2, 1, dim, 0, n_layers=2, act=torch.tanh)
        self.fw = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=dropout)
        return

    def forward(self, q, k, v=None, mask=None):
        """
            :param q: Tensor of (batch_size, hidden_size) or (batch_size, out_size, hidden_size), query, out_size = 1 if discarded
            :param k: Tensor of (batch_size, in_size, hidden_size), key
            :param v: Tensor of (batch_size, in_size, hidden_size), value, default = None, means v = k
            :param mask: Tensor of (batch_size, in_size), key/value mask, where 0 means data and 1 means pad, default = None, means zero matrix
            :returns: (output, attn)
                output: Tensor of (batch_size, hidden_size) or (batch_size, out_size, hidden_size), attention output, shape according to q
                attn: Tensor of (batch_size, in_size) or (batch_size, out_size, in_size), attention weight, shape according to q
        """
        if v is None:
            v = k
        q_dim = q.dim()
        if q_dim == 2:
            q = q.unsqueeze(1)
        output_size = q.size(1)
        input_size = k.size(1)
        qe = q.unsqueeze(2).expand(-1, -1, input_size, -1)
        ke = k.unsqueeze(1).expand(-1, output_size, -1, -1)
        score = self.score(torch.cat((qe, ke), dim=-1)).squeeze(-1)
        if mask is not None:
            score.masked_fill_(mask.unsqueeze(1).expand(-1, output_size, -1
                ), -float('inf'))
        attn = F.softmax(score, dim=-1)
        output = torch.bmm(attn, v)
        if q_dim == 2:
            output = output.squeeze(1)
            attn = attn.squeeze(1)
        output = F.leaky_relu(self.fw(self.dropout(output)))
        return output, attn


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'dropout': 0.5}]
