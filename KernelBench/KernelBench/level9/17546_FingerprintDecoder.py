import torch
import torch.utils.data
import torch.nn.functional as F


class FingerprintDecoder(torch.nn.Module):

    def __init__(self, n_in, n_out, dropout=0.1):
        super(FingerprintDecoder, self).__init__()
        if n_out > n_in:
            n_hidden = n_out // 2
        else:
            n_hidden = n_in // 2
        self.fc1 = torch.nn.Linear(n_in, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_out)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4, 'n_out': 4}]
