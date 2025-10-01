import torch


class TransformerNet2(torch.nn.Module):

    def __init__(self):
        super(TransformerNet2, self).__init__()
        self.tanh = torch.nn.Tanh()
        self.a = 10

    def forward(self, r, p):
        m = -0.5 * self.tanh(self.a * (p - 2 * r)) + 0.5 * self.tanh(self.a *
            (p - 2 * (1 - r)))
        return m


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
