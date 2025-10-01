import torch


class torch_return_int8_argmax(torch.nn.Module):

    def __init__(self):
        super(torch_return_int8_argmax, self).__init__()

    def forward(self, x):
        x0 = x.squeeze(0)
        _, x1 = torch.max(x0, 0)
        return x1


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
