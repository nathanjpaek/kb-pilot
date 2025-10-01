import torch


class Net(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden, n_hidden)
        self.layer4 = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        h_3 = sigmoid(self.layer3(h_2))
        out = self.layer4(h_3) ** 2
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_input': 4, 'n_hidden': 4, 'n_output': 4}]
