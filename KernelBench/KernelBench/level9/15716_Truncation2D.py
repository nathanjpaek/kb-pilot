import torch


class Truncation2D(torch.nn.Module):
    """
    A module merging the last two dimensions, merging coarse scale in grid
    of dimensions -4, -3 and finer resolution in dimensions -2, -1 to
    one fine grained grid with two dimensions less.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        """
        :param input: input tensor
        :returns: output tensor
        """
        shape = input.shape
        outputshape = list(input.shape[:-2])
        expsize1 = input.shape[-2]
        expsize2 = input.shape[-1]
        outputshape[-2] *= input.shape[-2]
        outputshape[-1] *= input.shape[-1]
        baseslice = [slice(None, None, 1) for _ in range(len(outputshape) - 2)]
        output = torch.zeros(outputshape, device=input.device,
            requires_grad=False)
        for i in range(shape[-4]):
            for j in range(shape[-3]):
                outslice = tuple(baseslice + [slice(expsize1 * i, expsize1 *
                    (i + 1)), slice(expsize2 * j, expsize2 * (j + 1))])
                inslice = tuple(baseslice + [i, j, slice(None, None, 1),
                    slice(None, None, 1)])
                output[outslice] += input[inslice]
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
