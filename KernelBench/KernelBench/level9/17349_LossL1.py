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


class LossL1(LossAbstract):
    """L1 distance loss."""

    def __init__(self, device='cuda:0'):
        super().__init__(device=device)
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, output, target):
        return self.l1(output, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
