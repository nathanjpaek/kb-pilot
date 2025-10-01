import torch
import torch.cuda
import torch.distributed


class FeedForward(torch.nn.Module):

    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, input_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(input_size)

    def forward(self, src):
        ret = self.linear1(self.norm(src))
        ret = self.linear2(self.dropout(torch.nn.functional.relu(ret)))
        return src + self.dropout(ret)

    def update_dropout(self, dropout):
        self.dropout.p = dropout


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'dropout': 0.5}]
