import math
import torch
from torch import nn


def apply_last_dim(model, x):
    size = list(x.size())
    y = model(x.contiguous().view(-1, size[-1]))
    size[-1] = y.size(-1)
    y = y.view(torch.Size(size))
    return y


def get_int_dim_index(name):
    if isinstance(name, int):
        return name
    name_list = 'axyz'
    assert name in name_list
    return [i for i in range(len(name_list)) if name_list[i] == name][0] - 1


class MinPoolTrinary(nn.Module):

    def __init__(self):
        super().__init__()

    def new_length(self, length):
        return length

    def forward(self, states):
        """
        :param states: [batch, length, *]
        """
        assert states.size(1) >= 3
        side_length = (states.size(1) + 1) // 3
        return torch.cat([torch.min(states[:, :side_length], dim=1, keepdim
            =True)[0], torch.min(states[:, side_length:-side_length], dim=1,
            keepdim=True)[0], torch.min(states[:, -side_length:], dim=1,
            keepdim=True)[0]], dim=1)

    def show(self, name='MinPoolTrinary', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(x) = MinPoolTrinary()' % (name,))


class Length(nn.Module):

    def __init__(self, dim_index=-1):
        super().__init__()
        self.dim_index = dim_index

    def forward(self, states, dim_index=None):
        if dim_index is None:
            dim_index = self.dim_index
        if isinstance(dim_index, int):
            dim_index = [dim_index]
        else:
            dim_index = [get_int_dim_index(x) for x in dim_index]
        if -1 in dim_index:

            def extractor(x):
                return torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
        else:

            def extractor(x):
                return torch.sqrt(torch.sum(x[:, dim_index].pow(2), dim=1,
                    keepdim=True))
        return apply_last_dim(extractor, states)

    def show(self, name='Length', indent=0, log=print, **kwargs):
        log(' ' * indent + "- %s(x) = |x's dim %s|" % (name, str(self.
            dim_index)))


class Distance(nn.Module):

    def __init__(self, dim_index=-1):
        super().__init__()
        self.dim_index = dim_index
        self.length = Length(dim_index)

    def forward(self, states1, states2, dim_index=None):
        return self.length(states1 - states2, dim_index)

    def show(self, name='Distance', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(x1, x2) = |x1 - x2|' % name)


class Position(nn.Module):

    def __init__(self, position_extractor=lambda x: x):
        super().__init__()
        self.position_extractor = position_extractor

    def forward(self, states):
        """
        :param states: [batch, length, n_agents, state_dim]
        """
        return apply_last_dim(self.position_extractor, states)

    def show(self, name='Position', indent=0, log=print, **kwargs):
        log(' ' * indent + "- %s(x) = x's first three dims" % name)


class SoftCompare(nn.Module):

    def __init__(self, alpha=None, beta=None):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * (0 if alpha is None else
            alpha), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1) * (0 if beta is None else
            beta), requires_grad=True)
        if alpha is None:
            nn.init.normal_(self.alpha.data, 0, 1)
        else:
            self.alpha.requires_grad_(False)
        if beta is not None:
            self.beta.requires_grad_(False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        raise NotImplementedError


class SoftSmall(SoftCompare):
    """
    Sigmoid((alpha - x) / e^beta)
    """

    def __init__(self, alpha=None, beta=None):
        super().__init__(alpha, beta)

    def forward(self, x, beta=None):
        alpha = self.alpha
        if beta is None:
            beta = self.beta
        return self.sigmoid((alpha - x) / torch.exp(beta))

    def show(self, name='SoftSmall', indent=0, log=print, **kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha
        beta = kwargs['beta'] if 'beta' in kwargs else self.beta
        log(' ' * indent + '- %s(x) = Sigmoid((%lf - x) / %lf)' % (name,
            alpha, math.exp(beta)))


class WithBall(nn.Module):

    def __init__(self, alpha=None, beta=None, trinary=False):
        super().__init__()
        self.position = Position()
        self.trinary = trinary
        self.distance = Distance()
        if trinary:
            self.min_pool_trinary = MinPoolTrinary()
        self.small = SoftSmall(alpha, beta)

    def new_length(self, length):
        return 3 if self.trinary else length

    def get_descriptions(self, n_agents, length):
        n_players = n_agents // 2
        agent_name = ['ball'] + [('A' + str(i)) for i in range(1, n_players +
            1)] + [('B' + str(i)) for i in range(1, n_players + 1)]
        res = []
        if self.trinary:
            trinary_name = ['pre', 'act', 'eff']
            for i in range(3):
                for p in range(1, n_agents):
                    res.append('WithBall(%s, %s)' % (agent_name[p],
                        trinary_name[i]))
        else:
            new_length = self.new_length(length)
            for i in range(0, new_length):
                for p in range(1, n_agents):
                    res.append('WithBall(%s, %.2f)' % (agent_name[p], (i + 
                        0.5) / new_length))
        return res

    def forward(self, states, beta=None):
        """
        :param states: [batch, length, n_agents, state_dim]
        return [batch, length, n_agents - 1, 1]
        """
        _batch, _length, _n_agents, _state_dim = states.size()
        ball_pos = self.position(states[:, :, :1])
        player_pos = self.position(states[:, :, 1:])
        dists = self.distance(player_pos, ball_pos)
        if self.trinary:
            dists = self.min_pool_trinary(dists)
        small = self.small(dists, beta=beta)
        return small

    def show(self, name='WithBall', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(p) = small(distance(p, ball))' % name)
        self.distance.show('distance', indent + 2, **kwargs)
        self.small.show('small', indent + 2, log=log, **kwargs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
