import torch
import torch.nn as nn


class SquadDiscriminator(nn.Module):

    def __init__(self, feature_size):
        super(SquadDiscriminator, self).__init__()
        self.bilinear = nn.Bilinear(feature_size, feature_size, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, global_enc, local_enc):
        global_enc = global_enc.unsqueeze(1)
        global_enc = global_enc.expand(-1, local_enc.size(1), -1)
        scores = self.bilinear(global_enc.contiguous(), local_enc.contiguous())
        return scores


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_size': 4}]
