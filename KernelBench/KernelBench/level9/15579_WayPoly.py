import torch


class WayPoly(torch.nn.Module):
    """Apply multiple modules to input and sum.

    It's equation for `poly_modules` length equal to :math:`N` could be expressed by

    !!!math

        I + F_1(I) + F_2(I) + ... + F_N

    where :math:`I` is identity and consecutive :math:`F_N` are consecutive `poly_modules`
    applied to input.

    Could be considered as an extension of standard `ResNet` to many parallel modules.

    Originally proposed by Xingcheng Zhang et al. in
    `PolyNet: A Pursuit of Structural Diversity in Very Deep Networks [here](https://arxiv.org/abs/1608.06993)

    Attributes:
        *poly_modules :
            Variable arg of modules to use. If empty, acts as an identity.
            For single module acts like `ResNet`. `2` was used in original paper.
            All modules need `inputs` and `outputs` of equal `shape`.
    """

    def __init__(self, *poly_modules: torch.nn.Module):
        """Initialize `WayPoly` object.
        
        Arguments:
            *poly_modules :
                Variable arg of modules to use. If empty, acts as an identity.
                For single module acts like `ResNet`. `2` was used in original paper.
                All modules need `inputs` and `outputs` of equal `shape`.
        """
        super().__init__()
        self.poly_modules: 'torch.nn.Module' = torch.nn.ModuleList(poly_modules
            )

    def forward(self, inputs):
        outputs = []
        for module in self.poly_modules:
            outputs.append(module(inputs))
        return torch.stack([inputs] + outputs, dim=0).sum(dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
