import torch
import torch.nn.functional as F


class LogSoftmax(torch.nn.Module):

    def __init__(self, dim):
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x, a):
        nll = -F.log_softmax(x, self.dim, _stacklevel=5)
        return (nll * a / a.sum(1, keepdim=True).clamp(min=1)).sum(dim=1).mean(
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
