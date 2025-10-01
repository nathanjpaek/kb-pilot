import torch


class FunctionalRelu(torch.nn.Module):

    def forward(self, x):
        return torch.nn.functional.relu(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
