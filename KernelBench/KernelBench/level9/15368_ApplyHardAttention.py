import torch


class ApplyHardAttention(torch.nn.Module):
    """
    ApplyHardAttention: Apply hard attention for the purpose of time-alignment.
    """

    def __init__(self):
        super().__init__()

    def forward(self, y, att):
        self.idx = att.argmax(2)
        y = y[torch.arange(y.shape[0]).unsqueeze(-1), self.idx]
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
