import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpBlock(nn.Module):

    def __init__(self, features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.features = features
        self.fc1 = nn.Linear(self.features, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, int(self.features))

    def forward(self, x):
        y = self.fc1(x)
        y = F.gelu(y)
        return self.fc2(y)


class MixerBlock(nn.Module):
    """Mixer block layer"""

    def __init__(self, tokens_dim, tokens_mlp_dim, channels_dim,
        channels_mlp_dim):
        super().__init__()
        self.tokens_mlp_dim = tokens_mlp_dim
        self.tokens_dim = tokens_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.channels_dim = channels_dim
        self.token_mixer = MlpBlock(self.tokens_dim, self.tokens_mlp_dim)
        self.channel_mixer = MlpBlock(self.channels_dim, self.channels_mlp_dim)

    def forward(self, x):
        y = F.layer_norm(x, x.shape[1:])
        y = torch.transpose(y, 1, 2)
        y = self.token_mixer(y)
        y = torch.transpose(y, 1, 2)
        x = x + y
        y = F.layer_norm(x, x.shape[1:])
        return x + self.channel_mixer(y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'tokens_dim': 4, 'tokens_mlp_dim': 4, 'channels_dim': 4,
        'channels_mlp_dim': 4}]
