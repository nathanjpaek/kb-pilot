import torch


def _tonemap(im):
    """Helper Reinhards tonemapper.
    Args:
        im(torch.Tensor): image to tonemap.
    Returns:
        (torch.Tensor) tonemaped image.
    """
    im = torch.clamp(im, min=0)
    return im / (1 + im)


class TonemappedRelativeMSE(torch.nn.Module):
    """Relative mean-squared error on tonemaped images.
    Args:
        eps(float): small number to avoid division by 0.
    """

    def __init__(self, eps=0.01):
        super(TonemappedRelativeMSE, self).__init__()
        self.eps = eps

    def forward(self, im, ref):
        im = _tonemap(im)
        ref = _tonemap(ref)
        mse = torch.pow(im - ref, 2)
        loss = mse / (torch.pow(ref, 2) + self.eps)
        loss = 0.5 * torch.mean(loss)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
