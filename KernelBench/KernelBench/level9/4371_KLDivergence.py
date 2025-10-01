import torch
import torch as th


class KLDivergence(th.nn.Module):
    """
    Args:
        min_value(float): the loss is clipped so that value below this
            number don't affect the optimization.
    """

    def __init__(self, min_value=0.2):
        super(KLDivergence, self).__init__()
        self.min_value = min_value

    def forward(self, mu, log_sigma):
        loss = -0.5 * (1.0 + log_sigma - mu.pow(2) - log_sigma.exp())
        loss = loss.mean()
        loss = th.max(loss, self.min_value * th.ones_like(loss))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
