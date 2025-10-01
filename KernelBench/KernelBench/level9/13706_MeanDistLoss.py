import torch


class MeanDistLoss(torch.nn.Module):

    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x, y):
        return torch.mean(torch.cdist(x, y, p=self.p))

    def extra_repr(self):
        return c_f.extra_repr(self, ['p'])


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
