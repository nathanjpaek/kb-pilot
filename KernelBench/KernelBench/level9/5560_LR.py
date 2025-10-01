import torch


class LR(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(LR, self).__init__()
        self.lr = torch.ones(input_size)
        self.lr = torch.nn.Parameter(self.lr)

    def forward(self, grad):
        return self.lr * grad


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
