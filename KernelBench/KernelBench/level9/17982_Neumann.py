import math
import torch
import torch.nn as nn


class Neumann(nn.Module):

    def __init__(self, n_features, depth, residual_connection, mlp_depth,
        init_type):
        super().__init__()
        self.depth = depth
        self.n_features = n_features
        self.residual_connection = residual_connection
        self.mlp_depth = mlp_depth
        self.relu = nn.ReLU()
        l_W = [torch.empty(n_features, n_features, dtype=torch.float) for _ in
            range(self.depth)]
        Wc = torch.empty(n_features, n_features, dtype=torch.float)
        beta = torch.empty(1 * n_features, dtype=torch.float)
        mu = torch.empty(n_features, dtype=torch.float)
        b = torch.empty(1, dtype=torch.float)
        l_W_mlp = [torch.empty(n_features, 1 * n_features, dtype=torch.
            float) for _ in range(mlp_depth)]
        l_b_mlp = [torch.empty(1 * n_features, dtype=torch.float) for _ in
            range(mlp_depth)]
        if init_type == 'normal':
            for W in l_W:
                nn.init.xavier_normal_(W)
            nn.init.xavier_normal_(Wc)
            nn.init.normal_(beta)
            nn.init.normal_(mu)
            nn.init.normal_(b)
            for W in l_W_mlp:
                nn.init.xavier_normal_(W)
            for b_mlp in l_b_mlp:
                nn.init.normal_(b_mlp)
        elif init_type == 'uniform':
            bound = 1 / math.sqrt(n_features)
            for W in l_W:
                nn.init.kaiming_uniform_(W, a=math.sqrt(5))
            nn.init.kaiming_uniform_(Wc, a=math.sqrt(5))
            nn.init.uniform_(beta, -bound, bound)
            nn.init.uniform_(mu, -bound, bound)
            nn.init.normal_(b)
            for W in l_W_mlp:
                nn.init.kaiming_uniform_(W, a=math.sqrt(5))
            for b_mlp in l_b_mlp:
                nn.init.uniform_(b_mlp, -bound, bound)
        self.l_W = [torch.nn.Parameter(W) for W in l_W]
        for i, W in enumerate(self.l_W):
            self.register_parameter('W_{}'.format(i), W)
        self.Wc = torch.nn.Parameter(Wc)
        self.beta = torch.nn.Parameter(beta)
        self.mu = torch.nn.Parameter(mu)
        self.b = torch.nn.Parameter(b)
        self.l_W_mlp = [torch.nn.Parameter(W) for W in l_W_mlp]
        for i, W in enumerate(self.l_W_mlp):
            self.register_parameter('W_mlp_{}'.format(i), W)
        self.l_b_mlp = [torch.nn.Parameter(b) for b in l_b_mlp]
        for i, b in enumerate(self.l_b_mlp):
            self.register_parameter('b_mlp_{}'.format(i), b)

    def forward(self, x, m, phase='train'):
        """
        Parameters:
        ----------
        x: tensor, shape (batch_size, n_features)
            The input data imputed by 0.
        m: tensor, shape (batch_size, n_features)
            The missingness indicator (0 if observed and 1 if missing).
        """
        h0 = x + m * self.mu
        h = x - (1 - m) * self.mu
        h_res = x - (1 - m) * self.mu
        if len(self.l_W) > 0:
            S0 = self.l_W[0]
            h = torch.matmul(h, S0) * (1 - m)
        for W in self.l_W[1:self.depth]:
            h = torch.matmul(h, W) * (1 - m)
            if self.residual_connection:
                h += h_res
        h = torch.matmul(h, self.Wc) * m + h0
        if self.mlp_depth > 0:
            for W, b in zip(self.l_W_mlp, self.l_b_mlp):
                h = torch.matmul(h, W) + b
                h = self.relu(h)
        y = torch.matmul(h, self.beta)
        y = y + self.b
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4, 'depth': 1, 'residual_connection': 4,
        'mlp_depth': 1, 'init_type': 4}]
