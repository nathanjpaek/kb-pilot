import torch


def _assert_no_grad(tensor):
    assert not tensor.requires_grad


class XSigmoid(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        _assert_no_grad(target)
        error = target - output
        return torch.mean(2 * error / (1 + torch.exp(-error)) - error)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
