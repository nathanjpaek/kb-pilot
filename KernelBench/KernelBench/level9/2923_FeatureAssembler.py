import torch
from typing import Optional
import torch.nn as nn


class FeatureAssembler(nn.Module):

    def __init__(self, T: 'int', embed_static: 'Optional[FeatureEmbedder]'=
        None, embed_dynamic: 'Optional[FeatureEmbedder]'=None) ->None:
        super().__init__()
        self.T = T
        self.embeddings = nn.ModuleDict({'embed_static': embed_static,
            'embed_dynamic': embed_dynamic})

    def forward(self, feat_static_cat: 'torch.Tensor', feat_static_real:
        'torch.Tensor', feat_dynamic_cat: 'torch.Tensor', feat_dynamic_real:
        'torch.Tensor') ->torch.Tensor:
        processed_features = [self.process_static_cat(feat_static_cat),
            self.process_static_real(feat_static_real), self.
            process_dynamic_cat(feat_dynamic_cat), self.
            process_dynamic_real(feat_dynamic_real)]
        return torch.cat(processed_features, dim=-1)

    def process_static_cat(self, feature: 'torch.Tensor') ->torch.Tensor:
        if self.embeddings['embed_static'] is not None:
            feature = self.embeddings['embed_static'](feature)
        return feature.unsqueeze(1).expand(-1, self.T, -1).float()

    def process_dynamic_cat(self, feature: 'torch.Tensor') ->torch.Tensor:
        if self.embeddings['embed_dynamic'] is None:
            return feature.float()
        else:
            return self.embeddings['embed_dynamic'](feature)

    def process_static_real(self, feature: 'torch.Tensor') ->torch.Tensor:
        return feature.unsqueeze(1).expand(-1, self.T, -1)

    def process_dynamic_real(self, feature: 'torch.Tensor') ->torch.Tensor:
        return feature


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4, 4]),
        torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'T': 4}]
