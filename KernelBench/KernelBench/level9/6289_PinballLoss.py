import torch
import torch.nn as nn


class PinballLoss(nn.Module):
    """Computes the pinball loss between y and y_hat.
  y: actual values in torch tensor.
  y_hat: predicted values in torch tensor.
  tau: a float between 0 and 1 the slope of the pinball loss. In the context
  of quantile regression, the value of alpha determine the conditional
  quantile level.
  return: pinball_loss
  """

    def __init__(self, tau=0.5):
        super(PinballLoss, self).__init__()
        self.tau = tau

    def forward(self, y, y_hat):
        delta_y = torch.sub(y, y_hat)
        pinball = torch.max(torch.mul(self.tau, delta_y), torch.mul(self.
            tau - 1, delta_y))
        pinball = pinball.mean()
        return pinball


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
