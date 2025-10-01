import torch
from torch import nn
import torch.nn.functional as F


class SetConv(nn.Module):

    def __init__(self, sample_feats, predicate_feats, join_feats,
        flow_feats, hid_units, num_hidden_layers=2):
        super(SetConv, self).__init__()
        self.flow_feats = flow_feats
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)
        if flow_feats > 0:
            self.flow_mlp1 = nn.Linear(flow_feats, hid_units)
            self.flow_mlp2 = nn.Linear(hid_units, hid_units)
            self.out_mlp1 = nn.Linear(hid_units * 4, hid_units)
        else:
            self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, samples, predicates, joins, flows, sample_mask,
        predicate_mask, join_mask):
        """
        #TODO: describe shapes
        """
        samples = samples
        predicates = predicates
        joins = joins
        sample_mask = sample_mask
        predicate_mask = predicate_mask
        join_mask = join_mask
        if self.flow_feats:
            flows = flows
            hid_flow = F.relu(self.flow_mlp1(flows))
            hid_flow = F.relu(self.flow_mlp2(hid_flow))
        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask
        hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
        sample_norm = sample_mask.sum(1, keepdim=False)
        hid_sample = hid_sample / sample_norm
        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        predicate_norm = predicate_mask.sum(1, keepdim=False)
        hid_predicate = hid_predicate / predicate_norm
        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        hid_join = hid_join * join_mask
        hid_join = torch.sum(hid_join, dim=1, keepdim=False)
        join_norm = join_mask.sum(1, keepdim=False)
        hid_join = hid_join / join_norm
        assert hid_sample.shape == hid_predicate.shape == hid_join.shape
        hid_sample = hid_sample.squeeze()
        hid_predicate = hid_predicate.squeeze()
        hid_join = hid_join.squeeze()
        if self.flow_feats:
            hid = torch.cat((hid_sample, hid_predicate, hid_join, hid_flow), 1)
        else:
            hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        return out


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]),
        torch.rand([4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]),
        torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'sample_feats': 4, 'predicate_feats': 4, 'join_feats': 4,
        'flow_feats': 4, 'hid_units': 4}]
