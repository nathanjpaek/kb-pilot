import torch


class CPUReverseForgetMult(torch.nn.Module):

    def __init__(self):
        super(CPUReverseForgetMult, self).__init__()

    def forward(self, f, x, hidden_init=None):
        result = []
        forgets = f.split(1, dim=0)[::-1]
        inputs = (f * x).split(1, dim=0)[::-1]
        prev_h = hidden_init
        for i, h in enumerate(inputs):
            h = h.squeeze()
            if prev_h is not None:
                h = h + (1 - forgets[i]) * prev_h
            result.append(h)
            prev_h = h
        result = result[::-1]
        return torch.cat(result, dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
