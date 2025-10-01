import torch
import torch as th


class SMAPE(th.nn.Module):
    """Symmetric Mean Absolute error.

    :math:`\\frac{|x - y|} {|x| + |y| + \\epsilon}`

    Args:
        eps(float): small number to avoid division by 0.
    """

    def __init__(self, eps=0.01):
        super(SMAPE, self).__init__()
        self.eps = eps

    def forward(self, im, ref):
        loss = (th.abs(im - ref) / (self.eps + th.abs(im.detach()) + th.abs
            (ref.detach()))).mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
