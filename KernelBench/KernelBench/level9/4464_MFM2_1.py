import torch


class MFM2_1(torch.nn.Module):
    """Max-Feature-Map (MFM) 2/1 operation. """

    def forward(self, input):
        input = input.reshape((input.shape[0], 2, -1, *input.shape[2:]))
        output = input.max(dim=1)[0]
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
