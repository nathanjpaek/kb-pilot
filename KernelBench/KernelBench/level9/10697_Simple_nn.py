import torch


class Simple_nn(torch.nn.Module):

    def __init__(self, dims_in, hidden):
        super(Simple_nn, self).__init__()
        self.linear1 = torch.nn.Linear(dims_in, hidden)
        self.linear2 = torch.nn.Linear(hidden, 2)
        self.output = torch.nn.LogSoftmax()

    def forward(self, x):
        hidden_activation = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(hidden_activation).clamp(min=0)
        return self.output(y_pred)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dims_in': 4, 'hidden': 4}]
