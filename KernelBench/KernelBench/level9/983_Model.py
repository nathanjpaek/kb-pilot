import torch


class Model(torch.nn.Module):

    def __init__(self, D_in, D_out):
        super(Model, self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn(D_in, D_out),
            requires_grad=True)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        y_pred = self.sig(x.mm(self.w1))
        return y_pred


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'D_in': 4, 'D_out': 4}]
