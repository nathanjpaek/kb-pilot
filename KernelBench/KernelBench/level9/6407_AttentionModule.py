import torch
from torch import nn


class AttentionModule(nn.Module):

    def __init__(self, feat_chans: 'int', state_chans: 'int',
        attention_units: 'int') ->None:
        super().__init__()
        self.feat_conv = nn.Conv2d(feat_chans, attention_units, 3, padding=1)
        self.state_conv = nn.Conv2d(state_chans, attention_units, 1, bias=False
            )
        self.attention_projector = nn.Conv2d(attention_units, 1, 1, bias=False)

    def forward(self, features: 'torch.Tensor', hidden_state: 'torch.Tensor'
        ) ->torch.Tensor:
        feat_projection = self.feat_conv(features)
        state_projection = self.state_conv(hidden_state)
        projection = torch.tanh(feat_projection + state_projection)
        attention = self.attention_projector(projection)
        attention = torch.flatten(attention, 1)
        attention = torch.softmax(attention, 1).reshape(-1, 1, features.
            shape[-2], features.shape[-1])
        glimpse = (features * attention).sum(dim=(2, 3))
        return glimpse


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feat_chans': 4, 'state_chans': 4, 'attention_units': 4}]
