import torch


class MLP(torch.nn.Module):

    def __init__(self, ni, no, nhidden, nlayers):
        super().__init__()
        self.nlayers = nlayers
        for i in range(nlayers):
            if i == 0:
                setattr(self, 'linear%d' % i, torch.nn.Linear(ni, nhidden,
                    bias=False))
            else:
                setattr(self, 'linear%d' % i, torch.nn.Linear(nhidden,
                    nhidden, bias=False))
            setattr(self, 'bn%d' % i, torch.nn.LayerNorm(nhidden))
        if nlayers == 0:
            nhidden = ni
        self.linear_out = torch.nn.Linear(nhidden, no)

    def forward(self, x):
        for i in range(self.nlayers):
            linear = getattr(self, 'linear%d' % i)
            bn = getattr(self, 'bn%d' % i)
            x = linear(x)
            x = bn(x)
            x = x * torch.sigmoid(x)
        return self.linear_out(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ni': 4, 'no': 4, 'nhidden': 4, 'nlayers': 1}]
