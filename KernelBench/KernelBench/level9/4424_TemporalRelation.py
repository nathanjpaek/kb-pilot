import torch
import torch.nn as nn


class TemporalRelation(nn.Module):

    def __init__(self, feat_dim, time_window=1):
        super(TemporalRelation, self).__init__()
        self.time_window = time_window
        self.feat_dim = feat_dim
        self.WT = nn.Linear(self.feat_dim, self.feat_dim, bias=False)

    def forward(self, feats):
        relation_feature = []
        att_feats = self.WT(feats)
        for t in range(0, att_feats.size()[0], 1):
            if t < self.time_window:
                prev = att_feats[0, :, :]
            else:
                prev = att_feats[t - 1, :, :]
            if t == att_feats.size()[0] - 1:
                next = att_feats[t, :, :]
            else:
                next = att_feats[t + 1, :, :]
            relation_feature.append(prev + next)
        relation_feature = torch.stack(relation_feature, dim=0)
        return relation_feature / 2 + feats


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feat_dim': 4}]
