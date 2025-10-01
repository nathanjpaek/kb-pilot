import torch


class RELUTwosided(torch.nn.Module):

    def __init__(self, num_conv, lam=0.001, L=100, sigma=1, device=None):
        super(RELUTwosided, self).__init__()
        self.L = L
        self.lam = torch.nn.Parameter(lam * torch.ones(1, num_conv, 1, 1,
            device=device))
        self.sigma = sigma
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        la = self.lam * self.sigma ** 2
        out = self.relu(torch.abs(x) - la / self.L) * torch.sign(x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_conv': 4}]
