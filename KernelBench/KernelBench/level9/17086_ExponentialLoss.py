import torch
from torch.nn.modules.loss import _Loss


class ExponentialLoss(_Loss):

    def __init__(self):
        super(ExponentialLoss, self).__init__()
        self.mseCriterion = torch.nn.modules.MSELoss()

    def forward(self, img, ref):
        return self.mseCriterion(img, ref) + 0.005 * self.mseCriterion(torch
            .exp(img), torch.exp(ref))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
