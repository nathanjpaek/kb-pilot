import torch
import torch.utils.data


class Vec2ArousalNet(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super(Vec2ArousalNet, self).__init__()
        self.layer_1 = torch.nn.Linear(D_in, H)
        self.layer_2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h = self.layer_1(x).clamp(min=0)
        y = self.layer_2(h).clamp(min=0, max=1)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'D_in': 4, 'H': 4, 'D_out': 4}]
