import torch


class Activation(torch.nn.Module):

    def __init__(self) ->None:
        super().__init__()

    def forward(self, inputs: 'torch.Tensor') ->torch.Tensor:
        raise NotImplementedError


class LeakyReLU(Activation):

    def forward(self, inputs: 'torch.Tensor') ->torch.Tensor:
        return torch.nn.functional.leaky_relu(inputs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
