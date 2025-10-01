import torch
import torch.utils.data


class ShapedSineModel(torch.nn.Module):

    def __init__(self, theta=None):
        super(ShapedSineModel, self).__init__()
        if theta is None:
            self.freq = torch.nn.Parameter(torch.Tensor([0.1]))
        else:
            self.freq = torch.nn.Parameter(torch.Tensor([theta]))
        self.learning_rate = 1.0

    def forward(self, x):
        return torch.sin(self.freq * x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
