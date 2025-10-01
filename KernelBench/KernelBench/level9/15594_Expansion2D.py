import torch


class Expansion2D(torch.nn.Module):
    """
    Expands a tensor in the last two dimensions, effectively to a coarse grid
    of smaller grids.
    """

    def __init__(self, expsize1: 'int', expsize2: 'int'):
        """
        :param expsize1: size of the second last dimension to be created
        :param expsize2: size of the last dimension to be created
        """
        super().__init__()
        self.expsize1 = expsize1
        self.expsize2 = expsize2

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        """
        :param input: input tensor
        :returns: output tensor
        """
        shape = list(input.shape)
        newshape = shape[:-2] + [shape[-2] // self.expsize1, shape[-1] //
            self.expsize2, self.expsize1, self.expsize2]
        sliceshape = list(newshape)
        sliceshape[-4] = 1
        sliceshape[-3] = 1
        output = torch.zeros(newshape, device=input.device)
        baseslice = [slice(None, None, 1) for _ in range(len(shape) - 2)]
        for i in range(shape[-2] // self.expsize1):
            for j in range(shape[-1] // self.expsize2):
                inslice = tuple(baseslice + [slice(self.expsize1 * i, self.
                    expsize1 * (i + 1)), slice(self.expsize2 * j, self.
                    expsize2 * (j + 1))])
                outslice = tuple(baseslice + [i, j, slice(None, None, 1),
                    slice(None, None, 1)])
                output[outslice] += input[inslice]
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'expsize1': 4, 'expsize2': 4}]
