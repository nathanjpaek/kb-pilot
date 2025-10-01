import torch


class MyActivation(torch.nn.Module):

    def __init__(self):
        super(MyActivation, self).__init__()
        self.relu = torch.nn.ReLU6(inplace=False)

    def forward(self, x):
        return x * self.relu(x + 3) / 6


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
