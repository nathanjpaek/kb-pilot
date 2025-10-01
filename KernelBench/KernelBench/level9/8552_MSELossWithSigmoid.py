import torch


class MSELossWithSigmoid(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.sigmoid = torch.nn.Sigmoid()
        self.loss = lambda x, y: self.mse(self.sigmoid(x), y)

    def forward(self, source, target):
        return self.loss(source, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
