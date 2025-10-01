import torch


class BaseModule(torch.nn.Module):

    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Highway(BaseModule):
    """
    Implementation as described
    in https://arxiv.org/pdf/1505.00387.pdf.
    """

    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = torch.nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = torch.nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)

    def forward(self, inputs):
        H = self.H(inputs).relu()
        T = self.T(inputs).sigmoid()
        outputs = H * T + inputs * (1 - T)
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'out_size': 4}]
