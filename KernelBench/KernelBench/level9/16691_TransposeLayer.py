import torch


class TransposeLayer(torch.nn.Module):
    """Transpose the input."""

    def forward(self, data):
        return data.t().contiguous()


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
