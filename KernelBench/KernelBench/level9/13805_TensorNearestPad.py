import torch


class TensorNearestPad(torch.nn.Module):

    def __init__(self, lower=1, upper=1):
        super().__init__()
        assert isinstance(lower, int) and lower >= 0
        assert isinstance(upper, int) and upper >= 0
        self.lower = lower
        self.upper = upper

    def forward(self, input):
        return torch.cat([input[:, :1].expand(-1, self.lower), input, input
            [:, -1:].expand(-1, self.upper)], dim=1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
