import torch
import torch.nn.functional as F


class Net_BP(torch.nn.Module):

    def __init__(self, n_features, n_hidden=50, n_output=1):
        super(Net_BP, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4}]
