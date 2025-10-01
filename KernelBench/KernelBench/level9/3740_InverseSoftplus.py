import torch
import torch.utils.data


def inverseSoftplus(y, beta=1, threshold=20):
    """
  inverse of y=torch.nn.functional.softplus(x, beta, threshold)
  :param y: the output of the softplus
  :param beta: the smoothness of the step
  :param threshold: the threshold after which a linear function is used
  :return: the input
  """
    return torch.where(y * beta > threshold, y, torch.log(torch.exp(beta *
        y) - 1) / beta)


class InverseSoftplus(torch.nn.Module):

    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self._beta = beta
        self._threshold = threshold

    def forward(self, y):
        return inverseSoftplus(y, self._beta, self._threshold)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
