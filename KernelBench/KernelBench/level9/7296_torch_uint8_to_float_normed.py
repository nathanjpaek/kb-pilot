import torch


class torch_uint8_to_float_normed(torch.nn.Module):

    def __init__(self):
        super(torch_uint8_to_float_normed, self).__init__()

    def forward(self, x):
        return (x.permute(2, 0, 1) / 255.0).unsqueeze(0).contiguous()


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
