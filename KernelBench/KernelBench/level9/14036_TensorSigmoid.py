import torch


class TensorSigmoid(torch.nn.Module):

    def __init__(self):
        super(TensorSigmoid, self).__init__()

    def forward(self, x):
        return x.sigmoid()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
