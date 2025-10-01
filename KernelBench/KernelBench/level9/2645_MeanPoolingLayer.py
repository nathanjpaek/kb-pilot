import torch


class BaseLayer(torch.nn.Module):

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MeanPoolingLayer(BaseLayer):

    def __init__(self):
        super(MeanPoolingLayer, self).__init__()

    def forward(self, input, dim=2):
        length = input.shape[2]
        return torch.sum(input, dim=2) / length


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
