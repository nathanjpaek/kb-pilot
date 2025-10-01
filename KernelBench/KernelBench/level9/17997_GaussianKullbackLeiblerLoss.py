import torch
import torch.nn as nn


class Loss(nn.Module):
    """Base loss class."""

    def __init__(self):
        super(Loss, self).__init__()


class GaussianKullbackLeiblerLoss(Loss):
    """Gaussian empirical KL divergence class."""

    def __init__(self) ->None:
        super(GaussianKullbackLeiblerLoss, self).__init__()

    def forward(self, P: 'torch.tensor', Q: 'torch.tensor') ->torch.tensor:
        """Kullback-Leibler divergence between two Gaussians.

        Args:
            P (torch.tensor): Tensor of reference model posterior parameter
                draws
            Q (torch.tensor): Tensor of submodel posterior parameter draws

        Returns:
            torch.tensor: Tensor of shape () containing sample KL divergence
        """
        loss = torch.mean(torch.abs(P - Q) ** 2) ** (1 / 2)
        assert loss.shape == (
            ), f'Expected data dimensions {()}, received {loss.shape}.'
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
