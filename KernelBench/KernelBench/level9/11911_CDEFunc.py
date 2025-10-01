import torch


class CDEFunc(torch.nn.Module):

    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)
        self.l1 = None
        self.l2 = None

    def forward(self, z):
        z = self.linear1(z)
        self.l1 = z
        z = z.relu()
        z = self.linear2(z)
        z = z.tanh()
        self.l2 = z
        z = z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        return z


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'hidden_channels': 4}]
