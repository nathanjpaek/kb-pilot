import torch
import torch.optim


class Threshold(torch.nn.Module):
    CAST_OPS = {'float': lambda t: t.float(), 'byte': lambda t: t.byte()}

    def __init__(self, value: 'float', comparison: 'str'='lower', dtype:
        'str'='float'):
        super(Threshold, self).__init__()
        self.threshold = value
        self.comp_op = (torch.le if comparison == 'lower' else torch.ge if 
            comparison == 'greater' else torch.ge)
        if dtype not in Threshold.CAST_OPS:
            log.error(
                'Casting operation type for Threshold monad should be either float or byte'
                )
        self.cast_op = Threshold.CAST_OPS[dtype]

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.cast_op(self.comp_op(x, self.threshold))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'value': 4}]
