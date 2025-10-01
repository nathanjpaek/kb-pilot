import torch
import torch as pt
import torch.distributed
import torch.distributed.elastic.multiprocessing.errors


class PyTorchLHUC(pt.nn.Module):
    """
    Learning Hidden Unit Contribution

    David Vilar. "Learning Hidden Unit Contribution for Adapting Neural
    Machine Translation Models" NAACL 2018

    :param num_hidden: Number of hidden units of the layer to be modified.
    """

    def __init__(self, num_hidden: 'int') ->None:
        super().__init__()
        self.weight = pt.nn.Parameter(pt.Tensor(num_hidden))

    def forward(self, data: 'pt.Tensor') ->pt.Tensor:
        weight = 2 * pt.sigmoid(self.weight)
        return weight * data

    def weights_from_mxnet_block(self, block_mx: "'LHUC'"):
        self.weight.data[:] = pt.as_tensor(block_mx.weight.data().asnumpy())


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_hidden': 4}]
