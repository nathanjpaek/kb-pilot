import torch
import torch.utils.data


def inverseSigmoid(y):
    """
  inverse of y=torch.sigmoid(y)
  :param y:
  :return: x
  """
    return torch.log(-y / (y - 1))


class InverseSigmoid(torch.nn.Module):

    def forward(self, y):
        return inverseSigmoid(y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
