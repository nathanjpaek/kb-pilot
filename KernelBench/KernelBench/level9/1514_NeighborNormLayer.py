import torch
import torch.nn as nn


class NeighborNormLayer(nn.Module):
    """Normalization layer that divides the output of a
    preceding layer by the number of neighbor features.
    Unlike the SimpleNormLayer, this layer allows for
    dynamically changing number of neighbors during training.
    """

    def __init__(self):
        super(NeighborNormLayer, self).__init__()

    def forward(self, input_features, n_neighbors):
        """Computes normalized output

        Parameters
        ----------
        input_features: torch.Tensor
            Input tensor of featuers of shape
            (n_frames, n_beads, n_feats)
        n_neighbors: int
            the number of neighbors

        Returns
        -------
        normalized_features: torch.Tensor
            Normalized input features
        """
        return input_features / n_neighbors


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
