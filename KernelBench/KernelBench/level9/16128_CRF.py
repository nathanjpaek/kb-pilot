import torch
from torch import nn
import torch.nn.init


class CRF(nn.Module):
    """
    Conditional Random Field.
    """

    def __init__(self, hidden_dim, tagset_size):
        """
        :param hidden_dim: size of word RNN/BLSTM's output
        :param tagset_size: number of tags
        """
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.emission = nn.Linear(hidden_dim, self.tagset_size)
        self.transition = nn.Parameter(torch.Tensor(self.tagset_size, self.
            tagset_size))
        self.transition.data.zero_()

    def forward(self, feats):
        """
        Forward propagation.

        :param feats: output of word RNN/BLSTM, a tensor of dimensions (batch_size, timesteps, hidden_dim)
        :return: CRF scores, a tensor of dimensions (batch_size, timesteps, tagset_size, tagset_size)
        """
        self.batch_size = feats.size(0)
        self.timesteps = feats.size(1)
        emission_scores = self.emission(feats)
        emission_scores = emission_scores.unsqueeze(2).expand(self.
            batch_size, self.timesteps, self.tagset_size, self.tagset_size)
        crf_scores = emission_scores + self.transition.unsqueeze(0).unsqueeze(0
            )
        return crf_scores


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4, 'tagset_size': 4}]
