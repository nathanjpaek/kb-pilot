import torch
import numpy as np


class SoftTree(torch.nn.Module):
    """Soft decision tree."""

    def __init__(self, in_features, out_features, depth, projection=
        'constant', dropout=0.0):
        super(SoftTree, self).__init__()
        self.proj = projection
        self.depth = depth
        self.in_features = in_features
        self.out_features = out_features
        self.leaf_count = int(2 ** depth)
        self.gate_count = int(self.leaf_count - 1)
        self.gw = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.
            empty(self.gate_count, in_features), nonlinearity='sigmoid').T)
        self.gb = torch.nn.Parameter(torch.zeros(self.gate_count))
        self.drop = torch.nn.Dropout(p=dropout)
        if self.proj == 'linear':
            self.pw = torch.nn.init.kaiming_normal_(torch.empty(
                out_features * self.leaf_count, in_features), nonlinearity=
                'linear')
            self.pw = torch.nn.Parameter(self.pw.view(out_features, self.
                leaf_count, in_features).permute(0, 2, 1))
            self.pb = torch.nn.Parameter(torch.zeros(out_features, self.
                leaf_count))
        elif self.proj == 'linear2':
            self.pw = torch.nn.init.kaiming_normal_(torch.empty(
                out_features * self.leaf_count, in_features), nonlinearity=
                'linear')
            self.pw = torch.nn.Parameter(self.pw.view(out_features, self.
                leaf_count, in_features).permute(1, 2, 0))
            self.pb = torch.nn.Parameter(torch.zeros(self.leaf_count, 1,
                out_features))
        elif self.proj == 'constant':
            self.z = torch.nn.Parameter(torch.randn(out_features, self.
                leaf_count))

    def forward(self, x):
        node_densities = self.node_densities(x)
        leaf_probs = node_densities[:, -self.leaf_count:].t()
        if self.proj == 'linear':
            gated_projection = torch.matmul(self.pw, leaf_probs).permute(2,
                0, 1)
            gated_bias = torch.matmul(self.pb, leaf_probs).permute(1, 0)
            result = torch.matmul(gated_projection, x.view(-1, self.
                in_features, 1))[:, :, 0] + gated_bias
        elif self.proj == 'linear2':
            x = x.view(1, x.shape[0], x.shape[1])
            out = torch.matmul(x, self.pw) + self.pb
            result = out, leaf_probs
        elif self.proj == 'constant':
            result = torch.matmul(self.z, leaf_probs).permute(1, 0)
        return result

    def extra_repr(self):
        return 'in_features=%d, out_features=%d, depth=%d, projection=%s' % (
            self.in_features, self.out_features, self.depth, self.proj)

    def node_densities(self, x):
        gw_ = self.drop(self.gw)
        gatings = torch.sigmoid(torch.add(torch.matmul(x, gw_), self.gb))
        node_densities = torch.ones(x.shape[0], 2 ** (self.depth + 1) - 1,
            device=x.device)
        it = 1
        for d in range(1, self.depth + 1):
            for i in range(2 ** d):
                parent_index = (it + 1) // 2 - 1
                child_way = (it + 1) % 2
                if child_way == 0:
                    parent_gating = gatings[:, parent_index]
                else:
                    parent_gating = 1 - gatings[:, parent_index]
                parent_density = node_densities[:, parent_index].clone()
                node_densities[:, it] = parent_density * parent_gating
                it += 1
        return node_densities

    def gatings(self, x):
        return torch.sigmoid(torch.add(torch.matmul(x, self.gw), self.gb))

    def total_path_value(self, x, index, level=None):
        gatings = self.gatings(x)
        gateways = np.binary_repr(index, width=self.depth)
        L = 0.0
        current = 0
        if level is None:
            level = self.depth
        for i in range(level):
            if int(gateways[i]) == 0:
                L += gatings[:, current].mean()
                current = 2 * current + 1
            else:
                L += (1 - gatings[:, current]).mean()
                current = 2 * current + 2
        return L


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4, 'depth': 1}]
