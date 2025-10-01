import torch
import torch.nn.functional as F


def mish(x):
    return x * torch.tanh(F.softplus(x))


class Net_mish_ranger(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net_mish_ranger, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden, n_output)
        self.predict = torch.nn.Linear(n_output, n_output)

    def forward(self, x):
        x = mish(self.hidden1(x))
        x = mish(self.hidden2(x))
        x = mish(self.hidden3(x))
        x = self.predict(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_feature': 4, 'n_hidden': 4, 'n_output': 4}]
