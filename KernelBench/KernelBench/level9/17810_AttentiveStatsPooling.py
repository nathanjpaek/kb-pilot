import torch
import torch.nn as nn


class AttentiveStatsPooling(nn.Module):
    """
    The attentive statistics pooling layer uses an attention
    mechanism to give different weights to different frames and
    generates not only weighted means but also weighted variances,
    to form utterance-level features from frame-level features

    "Attentive Statistics Pooling for Deep Speaker Embedding",
    Okabe et al., https://arxiv.org/abs/1803.10963
    """

    def __init__(self, input_size, hidden_size, eps=1e-06):
        super(AttentiveStatsPooling, self).__init__()
        self.eps = eps
        self.in_linear = nn.Linear(input_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, input_size)

    def forward(self, encodings):
        """
        Given encoder outputs of shape [B, DE, T], return
        pooled outputs of shape [B, DE * 2]

        B: batch size
        T: maximum number of time steps (frames)
        DE: encoding output size
        """
        energies = self.out_linear(torch.tanh(self.in_linear(encodings.
            transpose(1, 2)))).transpose(1, 2)
        alphas = torch.softmax(energies, dim=2)
        means = torch.sum(alphas * encodings, dim=2)
        residuals = torch.sum(alphas * encodings ** 2, dim=2) - means ** 2
        stds = torch.sqrt(residuals.clamp(min=self.eps))
        return torch.cat([means, stds], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
