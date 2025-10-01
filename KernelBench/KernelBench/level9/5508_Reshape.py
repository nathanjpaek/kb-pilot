import torch


class Reshape(torch.nn.Module):
    """
        Reshaping layer
    """

    def __init__(self, shapes1, shapes2):
        super(Reshape, self).__init__()
        self.shapes = shapes1, shapes2

    def forward(self, tensor):
        return torch.reshape(tensor.clone(), (tensor.shape[0], self.shapes[
            0], self.shapes[1]))


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'shapes1': 4, 'shapes2': 4}]
