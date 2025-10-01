import numpy
import torch


class HME(torch.nn.Module):

    def __init__(self, in_features, out_features, depth, projection='linear'):
        super(HME, self).__init__()
        self.proj = projection
        self.depth = depth
        self.in_features = in_features
        self.out_features = out_features
        self.n_leaf = int(2 ** depth)
        self.gate_count = int(self.n_leaf - 1)
        self.gw = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.
            empty(self.gate_count, in_features), nonlinearity='sigmoid').t())
        self.gb = torch.nn.Parameter(torch.zeros(self.gate_count))
        if self.proj == 'linear':
            self.pw = torch.nn.init.kaiming_normal_(torch.empty(
                out_features * self.n_leaf, in_features), nonlinearity='linear'
                )
            self.pw = torch.nn.Parameter(self.pw.reshape(out_features, self
                .n_leaf, in_features).permute(0, 2, 1))
            self.pb = torch.nn.Parameter(torch.zeros(out_features, self.n_leaf)
                )
        elif self.proj == 'constant':
            self.z = torch.nn.Parameter(torch.randn(out_features, self.n_leaf))

    def forward(self, x):
        node_densities = self.node_densities(x)
        leaf_probs = node_densities[:, -self.n_leaf:].t()
        if self.proj == 'linear':
            gated_projection = torch.matmul(self.pw, leaf_probs).permute(2,
                0, 1)
            gated_bias = torch.matmul(self.pb, leaf_probs).permute(1, 0)
            result = torch.matmul(gated_projection, x.reshape(-1, self.
                in_features, 1))[:, :, 0] + gated_bias
        elif self.proj == 'constant':
            result = torch.matmul(self.z, leaf_probs).permute(1, 0)
        return result

    def node_densities(self, x):
        gatings = self.gatings(x)
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

    def total_path_value(self, z, index, level=None):
        gatings = self.gatings(z)
        gateways = numpy.binary_repr(index, width=self.depth)
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

    def extra_repr(self):
        return 'in_features=%d, out_features=%d, depth=%d, projection=%s' % (
            self.in_features, self.out_features, self.depth, self.proj)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4, 'depth': 1}]
