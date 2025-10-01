import torch


class AddcmulTestModule(torch.nn.Module):

    def __init__(self, value):
        super(AddcmulTestModule, self).__init__()
        self.value = value

    def forward(self, x, y, z):
        return torch.addcmul(x, self.value, y, z)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'value': 4}]
