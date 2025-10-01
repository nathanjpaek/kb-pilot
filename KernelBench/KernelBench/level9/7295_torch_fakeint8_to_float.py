import torch


class torch_fakeint8_to_float(torch.nn.Module):

    def __init__(self):
        super(torch_fakeint8_to_float, self).__init__()

    def forward(self, x):
        x0 = x.permute(2, 0, 1)
        x0 += torch.clamp(x0, -1, 0) * -256.0
        return x0.unsqueeze(0).contiguous()


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
