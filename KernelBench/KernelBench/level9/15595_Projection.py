import torch
from typing import Tuple


class Projection(torch.nn.Module):
    """
    | A class for a projection of an input to a different shape     effectively mapping from
    | [..., inshape[1] .. inshape[-1]] -> [..., outshape[1] .. outshape[-1]]
    | only going over the subelements.
    | Example input (4,6) to (4,5) (shapes):
    | with instart (0, 1) inend (4, 5) outstart (0, 0), outend (4, 4)     maps essentially input[:, 1:5] to a new tensor output[:4, 0:4] with shape     (4, 5)
    | Non-indexed elements in the output are set to zero.

    """

    def __init__(self, instart: 'Tuple[int]', inend: 'Tuple[int]', inshape:
        'Tuple[int]', outstart: 'Tuple[int]', outend: 'Tuple[int]',
        outshape: 'Tuple[int]'):
        """
        :param instart: List of start indices of different dimensions in input
        :param inend: End indices (exclusive) in input
        :param inshape: Real input shapes (dimension sizes)
        :param outstart: List of start indices of different dimensions in output
        :param outend: End indices (exclusive) in output
        :param outshape: Real output shapes (dimension sizes)
        """
        super().__init__()
        self.inindex = tuple([slice(instart[i], inend[i], 1) for i in range
            (len(inshape))])
        self.outindex = tuple([slice(outstart[i], outend[i], 1) for i in
            range(len(outshape))])
        self.inshape = inshape
        self.outshape = outshape

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        """
        :param input: Input tensor
        :returns: output tensor
        """
        inindex = [slice(None, None, 1) for _ in range(len(input.shape) -
            len(self.inshape))]
        outindex = inindex
        inindex = tuple(inindex + list(self.inindex))
        outindex = tuple(outindex + list(self.outindex))
        outshape = [input.shape[i] for i in range(len(input.shape) - len(
            self.inshape))]
        outshape += self.outshape
        output = torch.zeros(outshape, device=input.device, requires_grad=False
            )
        output[outindex] += input[inindex]
        return output

    def backward(self, output: 'torch.Tensor') ->torch.Tensor:
        """
        :param output: output tensor to backward through module
        :returns: input gradient
        """
        outindex = [slice(None, None, 1) for _ in range(len(output.shape) -
            len(self.outshape))]
        inindex = outindex
        outindex = tuple(outindex + list(self.outindex))
        inindex = tuple(inindex + list(self.inindex))
        inshape = [output.shape[i] for i in range(len(output.shape) - len(
            self.inshape))]
        inshape += self.inshape
        input = torch.zeros(inshape, device=output.device, requires_grad=
            input.requires_grad)
        input[inindex] += output[outindex]
        return input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'instart': [4, 4], 'inend': [4, 4], 'inshape': [4, 4],
        'outstart': [4, 4], 'outend': [4, 4], 'outshape': [4, 4]}]
