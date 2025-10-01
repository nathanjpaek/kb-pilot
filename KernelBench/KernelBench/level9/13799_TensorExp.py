import torch


class TensorExp(torch.nn.Module):

    def forward(self, input):
        return torch.exp(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
