import torch


class ME(torch.nn.Module):

    def __init__(self, in_features, out_features, n_leaf, projection=
        'linear', dropout=0.0):
        super(ME, self).__init__()
        self.proj = projection
        self.n_leaf = n_leaf
        self.in_features = in_features
        self.out_features = out_features
        self.gw = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.
            empty(in_features, n_leaf)))
        self.gb = torch.nn.Parameter(torch.zeros(n_leaf))
        if self.proj == 'linear':
            self.pw = torch.nn.init.kaiming_normal_(torch.empty(
                out_features * n_leaf, in_features), nonlinearity='linear')
            self.pw = torch.nn.Parameter(self.pw.reshape(out_features,
                n_leaf, in_features).permute(0, 2, 1))
            self.pb = torch.nn.Parameter(torch.zeros(out_features, n_leaf))
        elif self.proj == 'constant':
            self.z = torch.nn.Parameter(torch.randn(out_features, n_leaf))

    def forward(self, x):
        gatings = torch.softmax(torch.add(torch.matmul(x, self.gw), self.gb
            ), dim=1).t()
        if self.proj == 'linear':
            gated_projection = torch.matmul(self.pw, gatings).permute(2, 0, 1)
            gated_bias = torch.matmul(self.pb, gatings).permute(1, 0)
            result = torch.matmul(gated_projection, x.reshape(-1, self.
                in_features, 1))[:, :, 0] + gated_bias
        elif self.proj == 'constant':
            result = torch.matmul(self.z, gatings).permute(1, 0)
        return result

    def extra_repr(self):
        return 'in_features=%d, out_features=%d, n_leaf=%d, projection=%s' % (
            self.in_features, self.out_features, self.n_leaf, self.proj)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4, 'n_leaf': 4}]
