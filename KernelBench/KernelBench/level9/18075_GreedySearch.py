import torch
import torch.nn as nn


def cuda():
    return torch.cuda.is_available()


def get_device():
    return torch.device('cuda' if cuda() else 'cpu')


class Search(nn.Module):
    """Base search class."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = get_device()

    def forward(self, logits: 'torch.Tensor') ->object:
        """
        Error handling.

        Args:
            logits: torch.Tensor (Tensor): the model's
                logits. (batch_size, length, vocabulary_size)
        Returns:
            object: the search output.
        """
        if not len(logits.shape) == 3:
            raise ValueError(
                f'Logits need to be 3D Tensor, was: {logits.shape}')
        if not type(logits) == torch.Tensor:
            raise TypeError(
                f'Logits need to be torch.Tensor, was: {type(logits)}')

    def step(self, logits: 'torch.Tensor') ->object:
        """
        Error handling.

        Args:
            logits: torch.Tensor (Tensor): the model's
                logits. (batch_size, vocabulary_size)
        Returns:
            object: the search output.
        """
        if len(logits.shape) > 3:
            raise ValueError(
                f'Logits need to be 2D or 3D Tensor, was: {logits.shape}')
        if not type(logits) == torch.Tensor:
            raise TypeError(
                f'Logits need to be torch.Tensor, was: {type(logits)}')


class GreedySearch(Search):
    """"Greedy search."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits: 'torch.Tensor') ->torch.Tensor:
        """
        Perform the greedy search.

        Args:
            logits: torch.Tensor (Tensor): the model's
                logits. (batch_size, length, vocabulary_size)
        Returns:
            torch.Tensor: the token indexes selected. (batch_size, length)
        """
        super().forward(logits)
        return torch.argmax(logits, 2)

    def step(self, logits: 'torch.Tensor') ->torch.Tensor:
        """
        Perform a greedy search step.

        Args:
            logits (torch.Tensor): the model's
                logits. (batch_size, vocabulary_size)
        Returns:
            torch.Tensor: the token indexes for all the batch. (batch_size, 1).
        """
        super().step(logits)
        return torch.argmax(logits, 1, keepdim=True)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
