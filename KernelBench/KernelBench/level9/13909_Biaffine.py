import torch
import torch.utils.data.dataloader
import torch.nn


class Biaffine(torch.nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        """
        :param n_in: size of input
        :param n_out: number of channels
        :param bias_x: set bias for x
        :param bias_x: set bias for y

        """
        super(Biaffine, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = torch.nn.Parameter(torch.Tensor(n_out, n_in + bias_x,
            n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        st = 'n_in:{}, n_out:{}, bias_x:{}, bias_x:{}'.format(self.n_in,
            self.n_out, self.bias_x, self.bias_y)
        return st

    def reset_parameters(self):
        torch.nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        s = s.squeeze(1)
        return s


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4}]
