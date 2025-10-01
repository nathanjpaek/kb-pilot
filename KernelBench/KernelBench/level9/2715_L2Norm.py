import torch


class L2Norm(torch.nn.Module):

    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        if len(norm.size()) == 1:
            x = x / norm.unsqueeze(-1).expand_as(x)
        else:
            [bs, _ch, h, w] = x.size()
            norm = norm.view(bs, 1, h, w)
            x = x / norm.expand_as(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
