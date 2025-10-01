import torch


class AdaptiveFeatureNorm(torch.nn.Module):
    """
    Implementation of the loss in
    [Larger Norm More Transferable:
    An Adaptive Feature Norm Approach for
    Unsupervised Domain Adaptation](https://arxiv.org/abs/1811.07456).
    Encourages features to gradually have larger and larger L2 norms.
    """

    def __init__(self, step_size: 'float'=1):
        """
        Arguments:
            step_size: The desired increase in L2 norm at each iteration.
                Note that the loss will always be equal to ```step_size```
                because the goal is always to make the L2 norm ```step_size```
                larger than whatever the current L2 norm is.
        """
        super().__init__()
        self.step_size = step_size

    def forward(self, x):
        """"""
        l2_norm = x.norm(p=2, dim=1)
        radius = l2_norm.detach() + self.step_size
        return torch.mean((l2_norm - radius) ** 2)

    def extra_repr(self):
        """"""
        return c_f.extra_repr(self, ['step_size'])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
