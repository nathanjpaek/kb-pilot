import torch


def sum_normalizer(x, detach=False, scale_by_batch_size=False):
    y = torch.sum(x)
    if detach:
        y = y.detach()
    if scale_by_batch_size:
        x = x * x.shape[0]
    return x / y


class SumNormalizer(torch.nn.Module):

    def __init__(self, detach=False, scale_by_batch_size=False):
        super().__init__()
        self.detach = detach
        self.scale_by_batch_size = scale_by_batch_size

    def forward(self, x):
        return sum_normalizer(x, detach=self.detach, scale_by_batch_size=
            self.scale_by_batch_size)

    def extra_repr(self):
        return c_f.extra_repr(self, ['detach', 'scale_by_batch_size'])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
