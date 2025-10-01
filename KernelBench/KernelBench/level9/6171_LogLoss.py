import torch
from torch.nn import MSELoss


class LogLoss(MSELoss):

    def __init__(self):
        super(LogLoss, self).__init__()
        self.loss = torch.nn.MSELoss()
        self.loss2 = torch.nn.MSELoss()

    def forward(self, input, target):
        tgt = torch.atan(target)
        inp = torch.atan(input)
        loss = torch.sqrt(self.loss(inp, tgt))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
