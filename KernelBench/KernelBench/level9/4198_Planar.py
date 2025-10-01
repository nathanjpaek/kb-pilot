import torch
import torch.nn as nn


class PlanarStep(nn.Module):

    def __init__(self):
        super(PlanarStep, self).__init__()
        self.h = nn.Tanh()
        self.softplus = nn.Softplus()

    def _der_h(self, x):
        """Derivative of activation function h."""
        return self._der_tanh(x)

    def _der_tanh(self, x):
        """Derivative of the Tanh function."""
        return 1 - self.h(x) ** 2

    def forward(self, zk, u, w, b):
        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b)
        or actually
        z'^T = z^T + h(z^T w + b)u^T
        Assumes the following input shapes:
        shape u = (batch_size, z_dim, 1)
        shape w = (batch_size, 1, z_dim)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_dim).
        """
        zk = zk.unsqueeze(2)
        uw = torch.bmm(w, u)
        m_uw = -1.0 + self.softplus(uw)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        u_hat = u + (m_uw - uw) * w.transpose(2, 1) / w_norm_sq
        wzb = torch.bmm(w, zk) + b
        z = zk + u_hat * self.h(wzb)
        z = z.squeeze(2)
        psi = w * self._der_h(wzb)
        logdet = torch.log(torch.abs(1 + torch.bmm(psi, u_hat)))
        logdet = logdet.squeeze(2).squeeze(1)
        return z, logdet


class Error(Exception):
    """Base error class, from which all other errors derive."""
    pass


class InvalidArgumentError(Error):
    """This error will be shown when a given argument has an invalid value."""
    pass


class NormalizingFlow(nn.Module):
    """Base class for normalizing flows."""

    def __init__(self, h_dim, z_dim, flow_depth, hidden_depth):
        super(NormalizingFlow, self).__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.flow_depth = flow_depth
        self.hidden_depth = hidden_depth

    @property
    def flow_depth(self):
        return self._flow_depth

    @flow_depth.setter
    def flow_depth(self, value):
        if not isinstance(value, int):
            raise InvalidArgumentError('flow_depth should be an integer.')
        elif value < 1:
            raise InvalidArgumentError(
                'flow_depth should be strictly positive.')
        else:
            self._flow_depth = value

    @property
    def hidden_depth(self):
        return self._hidden_depth

    @hidden_depth.setter
    def hidden_depth(self, value):
        if not isinstance(value, int):
            raise InvalidArgumentError('hidden_depth should be an integer.')
        elif value < 0:
            raise InvalidArgumentError('hidden_depth should be positive.')
        else:
            self._hidden_depth = value


class Planar(NormalizingFlow):
    """Planar Normalizing flow with single unit bottleneck."""

    def __init__(self, h_dim, z_dim, flow_depth):
        super(Planar, self).__init__(h_dim, z_dim, flow_depth, 0)
        self.flow = PlanarStep()
        self.h_to_u = nn.Linear(self.h_dim, self.flow_depth * self.z_dim)
        self.h_to_w = nn.Linear(self.h_dim, self.flow_depth * self.z_dim)
        self.h_to_b = nn.Linear(self.h_dim, self.flow_depth)

    def forward(self, z, h):
        u = self.h_to_u(h).view(-1, self.flow_depth, self.z_dim, 1)
        w = self.h_to_w(h).view(-1, self.flow_depth, 1, self.z_dim)
        b = self.h_to_b(h).view(-1, self.flow_depth, 1, 1)
        z_k = z
        logdet = 0.0
        for k in range(self.flow_depth):
            z_k, ldj = self.flow(z_k, u[:, k, :, :], w[:, k, :, :], b[:, k,
                :, :])
            logdet += ldj
        return z_k, logdet


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'h_dim': 4, 'z_dim': 4, 'flow_depth': 1}]
