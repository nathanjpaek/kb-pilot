import torch
import torch.nn as nn


class TripletLossDoubleMargin(nn.Module):
    """
    Triplet Loss with positive and negative margins, following the work of [1]
    
    References
    ----------
    [1] Ho, K., Keuper, J., Pfreundt, F. J., & Keuper, M. (2021, January). 
    Learning embeddings for image clustering: An empirical study of triplet loss approaches. 
    In 2020 25th International Conference on Pattern Recognition (ICPR) (pp. 87-94). IEEE.
    """

    def __init__(self, pos_margin=1.0, neg_margin=3.0):
        """
        Constructor of the loss.

        Parameters
        ----------
        pos_margin : float, optional
            Margin for positive examples. The default is 1.0.
        neg_margin : float, optional
            Margin for negative examples. The default is 3.0.

        Returns
        -------
        None.

        """
        super(TripletLossDoubleMargin, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor, positive, negative):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(self.neg_margin - distance_negative) + torch.relu(
            distance_positive - self.pos_margin)
        return losses.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
