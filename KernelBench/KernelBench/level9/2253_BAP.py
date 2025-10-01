import torch
import torch.nn as nn


class BAP(nn.Module):

    def __init__(self, **kwargs):
        super(BAP, self).__init__()

    def forward(self, feature_maps, attention_maps):
        feature_shape = feature_maps.size()
        attention_shape = attention_maps.size()
        phi_I = torch.einsum('imjk,injk->imn', (attention_maps, feature_maps))
        phi_I = torch.div(phi_I, attention_shape[1] * attention_shape[2])
        phi_I = torch.mul(torch.sign(phi_I), torch.sqrt(torch.abs(phi_I) + 
            1e-12))
        phi_I = phi_I.view(feature_shape[0], -1, 1, 1)
        raw_features = torch.nn.functional.normalize(phi_I, dim=-1)
        pooling_features = raw_features * 100.0
        return pooling_features


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
