from torch.nn import Module
import torch
from torch.nn.modules.module import Module


class AllocatingLayer(Module):
    """The actor NN base its output for the case of full CSI  on a continuous relaxation of the problem. Specifically it gives
    a value for every user. This layer will start allocating to the most valuable bw until no more resources are available for 
    the least valuable users
    """

    def __init__(self, Resource):
        super(AllocatingLayer, self).__init__()
        self.W = Resource

    def forward(self, values, weights):
        batchSize, Kusers = values.shape
        assert list(weights.size()) == [batchSize, Kusers] and (values >= 0
            ).all()
        VperW_diff = values.unsqueeze(dim=1).detach() - values.unsqueeze(dim=2
            ).detach()
        assert list(VperW_diff.shape) == [batchSize, Kusers, Kusers]
        Better_j_than_i = 1.0 * (VperW_diff >= 0)
        Satisfying_Constr = self.W - torch.matmul(Better_j_than_i, weights.
            unsqueeze(dim=2)).squeeze() >= 0
        assert list(Satisfying_Constr.shape) == [batchSize, Kusers]
        return Satisfying_Constr * weights


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'Resource': 4}]
