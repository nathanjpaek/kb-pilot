import torch


class SvmLoss(torch.nn.Module):

    def __init__(self):
        super(SvmLoss, self).__init__()

    def forward(self, decisions, targets):
        targets = targets.float() * 2 - 1
        projection_dist = 1 - targets * decisions
        margin = torch.max(torch.zeros_like(projection_dist), projection_dist)
        return margin.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
