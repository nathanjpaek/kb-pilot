import torch
import torch.cuda
import torch.distributed


class SimpleGate(torch.nn.Module):

    def __init__(self, dim):
        super(SimpleGate, self).__init__()
        self.gate = torch.nn.Linear(2 * dim, dim, bias=True)
        self.sig = torch.nn.Sigmoid()

    def forward(self, in1, in2):
        z = self.sig(self.gate(torch.cat((in1, in2), dim=-1)))
        return z * in1 + (1.0 - z) * in2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
