import torch


class HuberLoss(torch.nn.Module):

    def __init__(self, beta=0.3):
        self.beta = beta
        super(HuberLoss, self).__init__()

    def forward(self, suggested, target):
        errors = torch.abs(suggested - target)
        mask = errors < self.beta
        l2_errors = 0.5 * errors ** 2 / self.beta
        l1_errors = errors - 0.5 * self.beta
        combined_errors = mask * l2_errors + ~mask * l1_errors
        return combined_errors.mean(dim=0).sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
