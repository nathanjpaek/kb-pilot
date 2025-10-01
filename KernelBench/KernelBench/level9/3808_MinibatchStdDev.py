import torch


class MinibatchStdDev(torch.nn.Module):
    """
    Concatenate a constant statistic calculated across the minibatch to each pixel location (i, j) as a new channel.
    Here the standard deviation averaged over channels and locations. This is to increase variation of images produced
    by the generator. (see section 3)
    https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf
    """

    def __init__(self):
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, eps=1e-08):
        batch_size, _, img_width, img_height = x.shape
        y = x - x.mean(dim=0, keepdim=True)
        y = y.pow(2).mean(dim=0, keepdim=False).add(eps).sqrt()
        y = y.mean().view(1, 1, 1, 1)
        y = y.repeat(batch_size, 1, img_width, img_height)
        return torch.cat([x, y], 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
