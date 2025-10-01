import torch


class SoftL1(torch.nn.Module):

    def __init__(self):
        super(SoftL1, self).__init__()

    def forward(self, input, target, eps=0.0):
        l1 = torch.abs(input - target)
        ret = l1 - eps
        ret = torch.clamp(ret, min=0.0, max=100.0)
        return ret, torch.mean(l1.detach())


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
