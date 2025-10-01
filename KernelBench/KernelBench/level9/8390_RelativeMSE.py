import torch


class RelativeMSE(torch.nn.Module):
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
            im(torch.Tensor): image.
            ref(torch.Tensor): reference.
        """
        mse = torch.pow(im - ref, 2)
        loss = mse / (torch.pow(ref, 2) + self.eps)
        loss = 0.5 * torch.mean(loss)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
