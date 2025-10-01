import torch
import torch.nn as nn


class ACELoss(nn.Module):
    """
    Ref: [1] Aggregation Cross-Entropy for Sequence Recognition. CVPR-2019

    """

    def __init__(self, character, eps=1e-10):
        """
        Args:
            character (dict): recognition dictionary
            eps (float): margin of error
        """
        super(ACELoss, self).__init__()
        self.dict = character
        self.eps = eps

    def forward(self, inputs, label):
        """

        Args:
            inputs (Torch.Tensor): model output
            label (Torch.Tensor): label information

        Returns:
            Torch.Tensor: ace loss

        """
        batch, time_dim, _ = inputs.size()
        inputs = inputs + self.eps
        label = label.float()
        label[:, 0] = time_dim - label[:, 0]
        inputs = torch.sum(inputs, 1)
        inputs = inputs / time_dim
        label = label / time_dim
        loss = -torch.sum(torch.log(inputs) * label) / batch
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'character': 4}]
