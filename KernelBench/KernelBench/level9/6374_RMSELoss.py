import torch


class RMSELoss(torch.nn.Module):

    def __init__(self, eps=1e-08):
        super(RMSELoss, self).__init__()
        self.eps = eps
        self.criterion = torch.nn.MSELoss()

    def forward(self, y_hat, y):
        return torch.sqrt(self.criterion(y_hat, y) + self.eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
