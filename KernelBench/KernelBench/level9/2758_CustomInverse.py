import torch


class CustomTorchOp(torch.autograd.Function):

    @staticmethod
    def symbolic(g, input):
        return g.op('torchcustom::Add10', input)

    @staticmethod
    def forward(ctx, x):
        return x + 10


class CustomInverse(torch.nn.Module):

    def forward(self, x, y):
        ress = CustomTorchOp.apply(torch.inverse(x))
        return ress, torch.all(y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
