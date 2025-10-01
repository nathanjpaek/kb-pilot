import torch


class QuadraticModel(torch.nn.Module):

    def __init__(self, in_channels, class_num):
        super(QuadraticModel, self).__init__()
        x = torch.ones((in_channels, 1))
        self.x = torch.nn.parameter.Parameter(x.uniform_(-10.0, 10.0).float())

    def forward(self, A):
        return torch.sum(self.x * torch.matmul(A, self.x), -1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'class_num': 4}]
