import torch
import torch.nn as nn


class DisaggregatedPinballLoss(nn.Module):
    """ Pinball Loss
  Computes the pinball loss between y and y_hat.

  Parameters
  ----------
  y: tensor
    actual values in torch tensor.
  y_hat: tensor (same shape as y)
    predicted values in torch tensor.
  tau: float, between 0 and 1
    the slope of the pinball loss, in the context of
    quantile regression, the value of tau determines the
    conditional quantile level.

  Returns
  ----------
  pinball_loss:
    average accuracy for the predicted quantile
  """

    def __init__(self, tau=0.5):
        super(DisaggregatedPinballLoss, self).__init__()
        self.tau = tau

    def forward(self, y, y_hat):
        delta_y = torch.sub(y, y_hat)
        pinball = torch.max(torch.mul(self.tau, delta_y), torch.mul(self.
            tau - 1, delta_y))
        pinball = pinball.mean(axis=0).mean(axis=1)
        return pinball


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
