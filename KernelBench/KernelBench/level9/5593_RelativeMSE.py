import torch
import torch as th


class RelativeMSE(th.nn.Module):
    """Relative Mean-Squared Error.

    :math:`0.5 * \\frac{(x - y)^2}{y^2 + \\epsilon}`

    Args:
        eps(float): small number to avoid division by 0.
    """

    def __init__(self, eps=0.01):
        super(RelativeMSE, self).__init__()
        self.eps = eps

    def forward(self, im, ref):
        """Evaluate the metric.

        Args:
            im(th.Tensor): image.
            ref(th.Tensor): reference.
        """
        mse = th.pow(im - ref, 2)
        loss = mse / (th.pow(ref, 2) + self.eps)
        loss = 0.5 * th.mean(loss)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
