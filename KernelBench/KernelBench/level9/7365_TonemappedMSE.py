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


class TonemappedMSE(torch.nn.Module):
    """Mean-squared error on tonemaped images.
    Args:
        eps(float): small number to avoid division by 0.
    """

    def __init__(self, eps=0.01):
        super(TonemappedMSE, self).__init__()
        self.eps = eps

    def forward(self, im, ref):
        im = _tonemap(im)
        ref = _tonemap(ref)
        loss = torch.pow(im - ref, 2)
        loss = 0.5 * torch.mean(loss)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
