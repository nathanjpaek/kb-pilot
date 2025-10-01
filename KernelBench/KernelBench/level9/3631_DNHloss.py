import torch
import torch.nn as nn


class DNHloss(nn.Module):
    """DNH loss function

    Arguments:
        nn {[type]} -- [description]
    """

    def __init__(self, lamda):
        """Initializer class

        Arguments:
            lamda {[type]} -- [description]
        """
        super(DNHloss, self).__init__()
        self.lamda = lamda

    def forward(self, H, S):
        """Forward H and S

        Arguments:
            H {[type]} -- [description]
            S {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        theta = H @ H.t() / 2
        metric_loss = (torch.log(1 + torch.exp(-(self.lamda * theta).abs())
            ) + theta.clamp(min=0) - self.lamda * S * theta).mean()
        quantization_loss = self.logcosh(H.abs() - 1).mean()
        loss = metric_loss + self.lamda * quantization_loss
        return loss

    def logcosh(self, x):
        """log cos

        Arguments:
            x {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        return torch.log(torch.cosh(x))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'lamda': 4}]
