import torch
import torch.nn as nn


def module_test_print(var_input, var_inmed, var_ouput):
    for var in (var_input, var_inmed, var_ouput):
        None
        for key, value in var.items():
            None
            None


class PredLayer(nn.Module):

    def __init__(self, module_test=False):
        super(PredLayer, self).__init__()
        self.module_test = module_test

    def forward(self, u, v):
        """
        input:
            u: (batch_size, 1, dim_user)
            v: (batch_size, 1, dim_item)
        output:
            y: (batch_size, 1)   
        """
        _y = torch.mul(u, v)
        y = torch.sum(_y, dim=-1)
        if self.module_test:
            var_input = ['u', 'v']
            var_inmed = ['_y']
            var_ouput = ['y']
            locals_cap = locals()
            module_test_print(dict(zip(var_input, [eval(v, locals_cap) for
                v in var_input])), dict(zip(var_inmed, [eval(v, locals_cap) for
                v in var_inmed])), dict(zip(var_ouput, [eval(v, locals_cap) for
                v in var_ouput])))
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
