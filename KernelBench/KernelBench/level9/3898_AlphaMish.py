import torch


class AlphaMish(torch.nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros((in_features, 1, 1)))
        self.alpha.requires_grad = True

    def forward(self, x):
        return torch.mul(x, torch.tanh(torch.mul(1 + torch.nn.functional.
            softplus(self.alpha), torch.nn.functional.softplus(x))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4}]
