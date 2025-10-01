import torch
from torch import nn


class FeedForward(nn.Module):

    def __init__(self, ninp, dim_feedforward, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(ninp, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, ninp)
        self.norm1 = nn.LayerNorm(ninp)
        self.norm2 = nn.LayerNorm(ninp)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward_fn(self, x, branch):
        x = x + self.dropout1(branch)
        x = self.norm1(x)
        branch = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(branch)
        x = self.norm2(x)
        return x

    def forward(self, x, branch):
        chunk_size = 100000 // x.shape[2]
        outputs = []
        for i in range(0, x.shape[1], chunk_size):
            ed = min(i + chunk_size, x.shape[1])
            partial = self.forward_fn(x[:, i:ed], branch[:, i:ed])
            outputs.append(partial)
        return torch.cat(outputs, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ninp': 4, 'dim_feedforward': 4, 'dropout': 0.5}]
