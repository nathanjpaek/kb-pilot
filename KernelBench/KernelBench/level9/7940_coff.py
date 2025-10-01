import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class coff(nn.Module):

    def __init__(self, input_dims, fill_val=1, nl=None):
        super(coff, self).__init__()
        self.k = Parameter(torch.Tensor(1, input_dims))
        self.k.data.fill_(fill_val)
        self.nl = nn.Identity()
        if nl == 'sigmoid':
            self.nl = nn.Sigmoid()
        elif nl == 'tanh':
            self.nl = nn.Tanh()

    def forward(self, input):
        return self.nl(self.k) * input

    def extra_repr(self):
        return 'coff = {}'.format(self.nl(self.k).data.numpy().shape)

    @property
    def stats(self):
        return '%0.2f' % self.nl(self.k).detach().mean().cpu().numpy()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dims': 4}]
