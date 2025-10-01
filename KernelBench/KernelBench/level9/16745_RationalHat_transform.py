import torch
import torch.nn as nn


class RationalHat_transform(nn.Module):
    """
    Coordinate function as defined in 

    /Hofer, C., Kwitt, R., and Niethammer, M.
    Learning representations of persistence barcodes.
    JMLR, 20(126):1–45, 2019b./

    """

    def __init__(self, output_dim, input_dim=1):
        """
        output dim is the number of lines in the Line point transformation
        """
        super().__init__()
        self.output_dim = output_dim
        self.c_param = torch.nn.Parameter(torch.randn(input_dim, output_dim
            ) * 0.1, requires_grad=True)
        self.r_param = torch.nn.Parameter(torch.randn(1, output_dim) * 0.1,
            requires_grad=True)

    def forward(self, x):
        """
        x is of shape [N,input_dim]
        output is of shape [N,output_dim]
        """
        first_element = 1 + torch.norm(x[:, :, None] - self.c_param, p=1, dim=1
            )
        second_element = 1 + torch.abs(torch.abs(self.r_param) - torch.norm
            (x[:, :, None] - self.c_param, p=1, dim=1))
        return 1 / first_element - 1 / second_element


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'output_dim': 4}]
