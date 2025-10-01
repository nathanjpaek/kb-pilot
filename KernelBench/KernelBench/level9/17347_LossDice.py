import torch
import torch.nn as nn


class LossAbstract(nn.Module):
    """A named loss function, that loss functions should inherit from.
        Args:
            device (str): device key
    """

    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device
        self.name = self.__class__.__name__

    def get_evaluation_dict(self, output, target):
        """Return keys and values of all components making up this loss.
        Args:
            output (torch.tensor): a torch tensor for a multi-channeled model 
                output
            target (torch.tensor): a torch tensor for a multi-channeled target
        """
        return {self.name: float(self.forward(output, target).cpu())}


class LossDice(LossAbstract):
    """Dice loss with a smoothing factor."""

    def __init__(self, smooth=1.0, device='cuda:0'):
        super().__init__(device=device)
        self.smooth = smooth
        self.name = 'LossDice[smooth=' + str(self.smooth) + ']'

    def forward(self, output, target):
        output_flat = output.view(-1)
        target_flat = target.view(-1)
        intersection = (output_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (output_flat.sum() +
            target_flat.sum() + self.smooth)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
