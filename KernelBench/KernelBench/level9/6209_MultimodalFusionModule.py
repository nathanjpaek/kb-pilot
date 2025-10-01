import torch
import torch.nn as nn


class MultimodalFusionModule(nn.Module):

    def __init__(self, emb_dim, n_filters):
        super().__init__()
        self.fc_h = nn.Linear(emb_dim, n_filters)

    def forward(self, image, instruction):
        _batch_size, _n_filters, _height, _width = image.shape
        a = torch.sigmoid(self.fc_h(instruction))
        m = a.unsqueeze(-1).unsqueeze(-1)
        out = (m * image).contiguous()
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'emb_dim': 4, 'n_filters': 4}]
