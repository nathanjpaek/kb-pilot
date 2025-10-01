import torch


class GainesMul(torch.nn.Module):
    """
    this module is for Gaines stochastic multiplication, supporting unipolar/bipolar
    """

    def __init__(self, mode='bipolar', stype=torch.float):
        super(GainesMul, self).__init__()
        self.mode = mode
        self.stype = stype

    def UnaryMul_forward(self, input_0, input_1):
        if self.mode == 'unipolar':
            return input_0.type(torch.int8) & input_1.type(torch.int8)
        elif self.mode == 'bipolar':
            return 1 - (input_0.type(torch.int8) ^ input_1.type(torch.int8))
        else:
            raise ValueError('UnaryMul mode is not implemented.')

    def forward(self, input_0, input_1):
        return self.UnaryMul_forward(input_0, input_1).type(self.stype)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
