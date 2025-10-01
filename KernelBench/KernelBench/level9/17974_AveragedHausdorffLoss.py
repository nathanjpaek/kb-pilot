import torch
import torch.nn as nn


def cdist(x, y):
    """
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    """
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences ** 2, -1).sqrt()
    return distances


class AveragedHausdorffLoss(nn.Module):

    def __init__(self):
        super(AveragedHausdorffLoss, self).__init__()

    def forward(self, set1, set2):
        """
        Compute the Averaged Hausdorff Distance function
         between two unordered sets of points (the function is symmetric).
         Batches are not supported, so squeeze your inputs first!
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.
        """
        assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 2, 'got %s' % set2.ndimension()
        assert set1.size()[1] == set2.size()[1
            ], 'The points in both sets must have the same number of dimensions, got %s and %s.' % (
            set2.size()[1], set2.size()[1])
        d2_matrix = cdist(set1, set2)
        term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
        term_2 = torch.mean(torch.min(d2_matrix, 0)[0])
        res = term_1 + term_2
        return res


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
