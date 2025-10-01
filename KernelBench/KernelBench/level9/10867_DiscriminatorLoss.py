from torch.nn import Module
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn.functional
import torch.autograd


class DiscriminatorLoss(Module):
    """
    ## Discriminator Loss

    We want to find $w$ to maximize
    $$\\mathbb{E}_{x \\sim \\mathbb{P}_r} [f_w(x)]- \\mathbb{E}_{z \\sim p(z)} [f_w(g_	heta(z))]$$,
    so we minimize,
    $$-rac{1}{m} \\sum_{i=1}^m f_w ig(x^{(i)} ig) +
     rac{1}{m} \\sum_{i=1}^m f_w ig( g_	heta(z^{(i)}) ig)$$
    """

    def forward(self, f_real: 'torch.Tensor', f_fake: 'torch.Tensor'):
        """
        * `f_real` is $f_w(x)$
        * `f_fake` is $f_w(g_	heta(z))$

        This returns the a tuple with losses for $f_w(x)$ and $f_w(g_	heta(z))$,
        which are later added.
        They are kept separate for logging.
        """
        return F.relu(1 - f_real).mean(), F.relu(1 + f_fake).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
