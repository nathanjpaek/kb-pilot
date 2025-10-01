import torch


class MLP(torch.nn.Module):

    def __init__(self, insize, outsize=128, nonlinear=torch.nn.ReLU,
        activation=torch.nn.ReLU, hidden_layer_size=1, node_size=256):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential()
        self.net.add_module('fc_1', torch.nn.Linear(insize, node_size))
        self.net.add_module('nonlinear_1', nonlinear())
        for i in range(hidden_layer_size - 1):
            self.net.add_module('fc_' + str(i + 2), torch.nn.Linear(
                node_size, node_size))
            self.net.add_module('nonlinear_' + str(i + 2), nonlinear())
        self.net.add_module('head', torch.nn.Linear(node_size, outsize))
        self.net.add_module('activation', activation())

    def forward(self, x):
        return self.net(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'insize': 4}]
