import torch


class CustomInverse(torch.nn.Module):

    def forward(self, x, y):
        ress = torch.inverse(x) + x
        return ress, torch.all(y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
