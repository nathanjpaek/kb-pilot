import torch
import torch.nn as nn
import torch.utils.data
import torch.nn


class LearnMaskedDefault(nn.Module):
    """
    Learns default values to fill invalid entries within input tensors. The
    invalid entries are represented by a mask which is passed into forward alongside
    the input tensor. Note the default value is only used if all entries in the batch row are
    invalid rather than just a portion of invalid entries within each batch row.
    """

    def __init__(self, feature_dim: 'int', init_method: 'str'='gaussian',
        freeze: 'bool'=False):
        """
        Args:
            feature_dim (int): the size of the default value parameter, this must match the
                input tensor size.
            init_method (str): the initial default value parameter. Options:
                'guassian'
                'zeros'
            freeze (bool): If True, the learned default parameter weights are frozen.
        """
        super().__init__()
        if init_method == 'zeros':
            self._learned_defaults = nn.Parameter(torch.zeros(feature_dim),
                requires_grad=not freeze)
        elif init_method == 'gaussian':
            self._learned_defaults = nn.Parameter(torch.Tensor(feature_dim),
                requires_grad=not freeze)
            nn.init.normal_(self._learned_defaults)
        else:
            raise NotImplementedError(
                f"{init_method} not available. Options are: 'zeros' or 'gaussian'"
                )

    def forward(self, x: 'torch.Tensor', mask: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): tensor of shape (batch_size, feature_dim).
            mask (torch.Tensor): bool tensor of shape (batch_size, seq_len) If all elements
                in the batch dimension are False the learned default parameter is used for
                that batch element.

        Returns:
            Tensor with shape (batch_size, feature_dim)
        """
        mask = mask.view(mask.shape[0], -1).any(dim=-1)
        for i in range(1, x.dim()):
            mask = mask.unsqueeze(i)
        x = x * mask.float() + self._learned_defaults * (1 - mask.float())
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_dim': 4}]
