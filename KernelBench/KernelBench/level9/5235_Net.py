import torch


class Net(torch.nn.Module):
    """Implementing two layer nn."""

    def __init__(self, D_IN, H, D_OUT):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_IN, H)
        self.linear2 = torch.nn.Linear(H, D_OUT)

    def forward(self, x):
        h = self.linear1(x)
        h_relu = torch.clamp(h, 0)
        y_pred = self.linear2(h_relu)
        return y_pred


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'D_IN': 4, 'H': 4, 'D_OUT': 4}]
