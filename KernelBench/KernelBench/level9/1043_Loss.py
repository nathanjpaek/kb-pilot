import torch
import torch as t
import torch.nn as nn


def indicator(K):
    """
    @K:             number of users
    """
    return t.eye(5 * K)


class Loss(nn.Module):

    def __init__(self, K, Nt, Vartheta):
        super(Loss, self).__init__()
        self.K = K
        self.Nt = Nt
        self.Delta = indicator(self.K)
        self.alpha = 1 / self.K
        self.Vartheta = Vartheta
        self.batchsize = 10

    def forward(self, x, ind1, ind2):
        """
        @x:           output of the last layer, its dimmension is (batchsize, 2*K*K+3*K)
        
        """
        loss = self.alpha * t.mean(t.matmul(t.log(1 + x), ind1) - t.matmul(
            self.Vartheta * x, ind2))
        return -loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'K': 4, 'Nt': 4, 'Vartheta': 4}]
