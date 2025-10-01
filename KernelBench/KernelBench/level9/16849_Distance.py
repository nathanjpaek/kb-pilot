import torch
from torch import nn


def apply_last_dim(model, x):
    size = list(x.size())
    y = model(x.contiguous().view(-1, size[-1]))
    size[-1] = y.size(-1)
    y = y.view(torch.Size(size))
    return y


def get_int_dim_index(name):
    if isinstance(name, int):
        return name
    name_list = 'axyz'
    assert name in name_list
    return [i for i in range(len(name_list)) if name_list[i] == name][0] - 1


class Length(nn.Module):

    def __init__(self, dim_index=-1):
        super().__init__()
        self.dim_index = dim_index

    def forward(self, states, dim_index=None):
        if dim_index is None:
            dim_index = self.dim_index
        if isinstance(dim_index, int):
            dim_index = [dim_index]
        else:
            dim_index = [get_int_dim_index(x) for x in dim_index]
        if -1 in dim_index:

            def extractor(x):
                return torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
        else:

            def extractor(x):
                return torch.sqrt(torch.sum(x[:, dim_index].pow(2), dim=1,
                    keepdim=True))
        return apply_last_dim(extractor, states)

    def show(self, name='Length', indent=0, log=print, **kwargs):
        log(' ' * indent + "- %s(x) = |x's dim %s|" % (name, str(self.
            dim_index)))


class Distance(nn.Module):

    def __init__(self, dim_index=-1):
        super().__init__()
        self.dim_index = dim_index
        self.length = Length(dim_index)

    def forward(self, states1, states2, dim_index=None):
        return self.length(states1 - states2, dim_index)

    def show(self, name='Distance', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(x1, x2) = |x1 - x2|' % name)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
