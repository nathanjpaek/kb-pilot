import torch


class L0Loss(torch.nn.Module):

    def forward(self, suggested, target):
        errors = (suggested - target).abs()
        return torch.max(errors, dim=-1)[0].mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
