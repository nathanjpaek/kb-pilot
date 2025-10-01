import torch


class LossLoglikelihoodNb(torch.nn.Module):

    def __init__(self, average=True):
        super(LossLoglikelihoodNb, self).__init__()
        self.average = average

    def forward(self, preds, target):
        """Implements the negative log likelihood loss as VAE reconstruction loss"""
        x = target
        loc, scale = torch.chunk(preds, chunks=2, dim=1)
        eta_loc = torch.log(loc)
        eta_scale = torch.log(scale)
        log_r_plus_mu = torch.log(scale + loc)
        ll = torch.lgamma(scale + x)
        ll = ll - torch.lgamma(x + torch.ones_like(x))
        ll = ll - torch.lgamma(scale)
        ll = ll + torch.multiply(x, eta_loc - log_r_plus_mu) + torch.multiply(
            scale, eta_scale - log_r_plus_mu)
        ll = torch.clamp(ll, min=-300, max=300)
        neg_ll = -ll
        if self.average:
            neg_ll = torch.mean(neg_ll)
        else:
            neg_ll = neg_ll.sum(dim=1).sum(dim=1)
        return neg_ll


def get_inputs():
    return [torch.rand([4, 2, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
