import torch


class ConstantODE(torch.nn.Module):

    def __init__(self):
        super(ConstantODE, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.2))
        self.b = torch.nn.Parameter(torch.tensor(3.0))

    def forward(self, t, y):
        return self.a + (y - (self.a * t + self.b)) ** 5

    def y_exact(self, t):
        return self.a * t + self.b


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
