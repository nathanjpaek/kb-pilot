import torch


class SimpleLoss(torch.nn.Module):

    def forward(self, output, target):
        return output / target


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
