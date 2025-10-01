import torch


class fromImageToTensor(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        tensor = tensor.float() / 255.0
        return tensor


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
