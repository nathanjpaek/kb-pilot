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


class MlpMixer(nn.Module):
    """Mixer architecture"""

    def __init__(self, patches, feature_length, num_classes, num_blocks,
        hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.patches = patches
        self.feature_length = feature_length
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.conv = nn.Conv1d(in_channels=self.patches, out_channels=self.
            hidden_dim, kernel_size=1)
        for nb in range(self.num_blocks):
            setattr(self, 'mixerBlock_{}'.format(nb), MixerBlock(self.
                feature_length, self.tokens_mlp_dim, self.hidden_dim, self.
                channels_mlp_dim))
        self.fc = nn.Linear(in_features=self.feature_length, out_features=
            self.num_classes)

    def forward(self, inputs):
        x = torch.transpose(inputs, 1, 2)
        x = self.conv(x)
        x = torch.transpose(x, 1, 2)
        for nb in range(self.num_blocks):
            x = getattr(self, 'mixerBlock_{}'.format(nb))(x)
        x = F.layer_norm(x, x.shape[1:])
        x = torch.transpose(x, 1, 2)
        x = torch.mean(x, dim=1)
        return self.fc(x)


class OrMixer(nn.Module):

    def __init__(self, n_layers, entity_dim):
        super(OrMixer, self).__init__()
        self.mlpMixer = MlpMixer(patches=2, feature_length=entity_dim,
            num_classes=entity_dim, num_blocks=3, hidden_dim=20,
            tokens_mlp_dim=200, channels_mlp_dim=4)

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(-1)
        x2 = x2.unsqueeze(-1)
        x = torch.cat((x1, x2), dim=-1)
        x = self.mlpMixer(x)
        return x


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_layers': 1, 'entity_dim': 4}]
