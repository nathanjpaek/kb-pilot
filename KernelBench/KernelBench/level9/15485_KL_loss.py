import torch
import torch.nn.functional


class KL_loss(torch.nn.Module):

    def __init__(self):
        super(KL_loss, self).__init__()

    def forward(self, mu, logvar):
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar
            )
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return KLD


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
