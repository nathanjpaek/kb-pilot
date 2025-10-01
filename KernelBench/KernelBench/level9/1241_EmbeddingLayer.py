import torch
import torch.nn.functional


class EmbeddingLayer(torch.nn.Module):
    """Attention layer."""

    def __init__(self, feature_number: 'int'):
        """Initialize the relational embedding layer.

        :param feature_number: Number of features.
        """
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(feature_number,
            feature_number))
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(self, left_representations: 'torch.FloatTensor',
        right_representations: 'torch.FloatTensor', alpha_scores:
        'torch.FloatTensor'):
        """
        Make a forward pass with the drug representations.

        :param left_representations: Left side drug representations.
        :param right_representations: Right side drug representations.
        :param alpha_scores: Attention scores.
        :returns: Positive label scores vector.
        """
        attention = torch.nn.functional.normalize(self.weights, dim=-1)
        left_representations = torch.nn.functional.normalize(
            left_representations, dim=-1)
        right_representations = torch.nn.functional.normalize(
            right_representations, dim=-1)
        attention = attention.view(-1, self.weights.shape[0], self.weights.
            shape[1])
        scores = alpha_scores * (left_representations @ attention @
            right_representations.transpose(-2, -1))
        scores = scores.sum(dim=(-2, -1)).view(-1, 1)
        return scores


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_number': 4}]
