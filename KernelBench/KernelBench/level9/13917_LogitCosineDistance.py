import torch
import torch.utils.data.dataloader
import torch.nn


def dot_product(a: 'torch.Tensor', b: 'torch.Tensor', normalize=False):
    """
    Computes dot product for pairs of vectors.
    :param normalize: Vectors are normalized (leads to cosine similarity)
    :return: Matrix with res[i][j]  = dot_product(a[i], b[j])
    """
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    if normalize:
        a = torch.nn.functional.normalize(a, p=2, dim=1)
        b = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a, b.transpose(0, 1))


class LogitCosineDistance(torch.nn.Module):

    def forward(self, a, b):
        return torch.logit(0.5 - 0.5 * dot_product(a, b, normalize=True))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
