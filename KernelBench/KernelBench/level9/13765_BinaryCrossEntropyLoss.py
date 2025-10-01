from torch.nn import Module
import torch
from torch import zeros_like
from torch import ones_like
from torch.nn import Sigmoid
from torch.nn import BCELoss


class BinaryCrossEntropyLoss(Module):
    """This class implements :class:`torch.nn.Module` interface.

    """

    def __init__(self):
        super().__init__()
        self.sig = Sigmoid()
        self.loss = BCELoss(reduction='sum')

    def forward(self, positive_triplets, negative_triplets):
        """

        Parameters
        ----------
        positive_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scores of the true triplets as returned by the `forward` methods
            of the models.
        negative_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scores of the negative triplets as returned by the `forward`
            methods of the models.
        Returns
        -------
        loss: torch.Tensor, shape: (n_facts, dim), dtype: torch.float
            Loss of the form :math:`-\\eta \\cdot \\log(f(h,r,t)) +
            (1-\\eta) \\cdot \\log(1 - f(h,r,t))` where :math:`f(h,r,t)`
            is the score of the fact and :math:`\\eta` is either 1 or
            0 if the fact is true or false.
        """
        return self.loss(self.sig(positive_triplets), ones_like(
            positive_triplets)) + self.loss(self.sig(negative_triplets),
            zeros_like(negative_triplets))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
