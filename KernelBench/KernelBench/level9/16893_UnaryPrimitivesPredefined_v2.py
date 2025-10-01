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


class Normalize(nn.Module):

    def __init__(self, distribution=None, **kwargs):
        super().__init__()
        self.distribution = distribution
        self.data_ = []
        if distribution is None:
            pass
        elif distribution == 'normal':
            mean = kwargs['mean'] if 'mean' in kwargs else 0
            std = kwargs['std'] if 'std' in kwargs else 1
            self.param = nn.Parameter(torch.Tensor([mean, std]), False)
        elif distribution == 'uniform':
            vmin = kwargs['minv'] if 'minv' in kwargs else 0
            vmax = kwargs['maxv'] if 'maxv' in kwargs else 1
            self.param = nn.Parameter(torch.Tensor([vmin, vmax]), False)
        else:
            raise NotImplementedError()

    def forward(self, x, keep_data=False):
        if keep_data:
            self.data_.append(x.detach().cpu().view(-1))
            return x
        if self.distribution is None:
            return x
        elif self.distribution == 'normal':
            mean = self.param[0]
            std = self.param[1]
            return (x - mean) / std
        elif self.distribution == 'uniform':
            vmin = self.param[0]
            vmax = self.param[1]
            return (x - vmin) / (vmax - vmin + 1e-05)
        else:
            raise NotImplementedError()

    def reset_parameters(self, name=None):
        assert len(self.data_) > 0
        data = torch.cat(self.data_)
        self.data_ = []
        if self.distribution is None:
            pass
        elif self.distribution == 'normal':
            with torch.no_grad():
                self.param[0] = data.mean().item()
                self.param[1] = data.std().item()
            if name is not None:
                None
        elif self.distribution == 'uniform':
            with torch.no_grad():
                self.param[0] = data.min().item()
                self.param[1] = data.max().item()
            if name is not None:
                None
        else:
            raise NotImplementedError()

    def recover_threshold(self, x):
        if self.distribution is None:
            return x
        elif self.distribution == 'normal':
            return x * float(self.param[1]) + float(self.param[0])
        elif self.distribution == 'uniform':
            return x * float(self.param[1] - self.param[0] + 1e-05) + float(
                self.param[0])
        else:
            raise NotImplementedError()

    def init_thresholds(self, x):
        if self.distribution is None:
            nn.init.normal_(x, 0, 1)
        elif self.distribution == 'normal':
            nn.init.normal_(x, 0, 1)
        elif self.distribution == 'uniform':
            nn.init.uniform_(x, 0, 1)
        else:
            raise NotImplementedError()


class SoftCmp(nn.Module):
    """
    Sigmoid((x - y) / e^beta)
    """

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, beta):
        return self.sigmoid((x - y) / math.exp(beta))


class Inequality(nn.Module):

    def __init__(self, out_dim=1, distribution=None, **kwargs):
        super().__init__()
        self.out_dim = out_dim
        self.thresholds = nn.Parameter(torch.zeros(out_dim), requires_grad=True
            )
        self.distribution = distribution
        self.normalize = Normalize(distribution)
        self.cmp = SoftCmp()
        self.normalize.init_thresholds(self.thresholds)

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_agents, ... ]
        """
        states_expand = states.view(*(states.size() + (1,)))
        estimate_parameters = 'estimate_parameters' in kwargs and kwargs[
            'estimate_parameters']
        states_expand = self.normalize(states_expand, keep_data=
            estimate_parameters)
        return self.cmp(states_expand, self.thresholds.view(*([1] * len(
            states.size()) + [self.out_dim])), beta)

    def reset_parameters(self, parameter_name, name=None):
        if parameter_name == 'primitive_inequality':
            self.normalize.reset_parameters(name=name)
            self.normalize.init_thresholds(self.thresholds)

    def get_descriptions(self, name='Inequality'):
        theta = self.thresholds.detach().cpu().view(self.out_dim)
        descroptions = []
        for k in range(theta.size(0)):
            t = self.normalize.recover_threshold(theta[k])
            if 'speed' in name:
                t = t * 8
            if 'acc' in name:
                t = t * 64
            descroptions.append('%s > %.2lf' % (name, t))
        return descroptions


class N_aryPrimitivesPredefined(nn.Module):

    def __init__(self):
        super().__init__()
        self.out_dim = 0
        self.primitive_list = []
        self.ineqs = nn.ModuleDict({})

    def reset_parameters(self, parameter_name):
        for k in self.primitive_list:
            self.ineqs[k].reset_parameters(parameter_name, name=k)

    def get_descriptions(self):
        descriptions = []
        for k in self.primitive_list:
            descriptions += self.ineqs[k].get_descriptions(name=k)
        return descriptions


class AlignDifferential(nn.Module):

    def __init__(self):
        super().__init__()

    def new_length(self, length):
        return length

    def forward(self, states):
        """
        :param states: [batch, length, *]
        """
        padded_states = torch.cat([states[:, 0:1] * 2 - states[:, 1:2],
            states, states[:, -1:] * 2 - states[:, -2:-1]], dim=1)
        return (padded_states[:, 2:] - padded_states[:, :-2]) / 2

    def show(self, name='AlignDifferential', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(x) = AlignDifferential()' % (name,))


class UnaryPrimitivesPredefined_v2(N_aryPrimitivesPredefined):

    def __init__(self, cmp_dim=10):
        super().__init__()
        self.differential = AlignDifferential()
        self.primitive_list = ['acc', 'pos_z', 'speed', 'dist_to_ball']
        self.distance = Distance()
        self.ineqs.update({'acc': Inequality(out_dim=cmp_dim, distribution=
            'normal'), 'pos_z': Inequality(out_dim=cmp_dim, distribution=
            'uniform'), 'speed': Inequality(out_dim=cmp_dim, distribution=
            'normal'), 'dist_to_ball': Inequality(out_dim=cmp_dim,
            distribution='normal')})
        self.out_dim = sum([self.ineqs[k].out_dim for k in self.primitive_list]
            )

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_agents, state_dim]
        return [batch, length, n_agents, out_dim]
        """
        velocity = self.differential(states)
        acc = self.differential(velocity)
        n_agents = states.size(2)
        p1 = states.unsqueeze(2).repeat(1, 1, n_agents, 1, 1)
        p2 = states.unsqueeze(3).repeat(1, 1, 1, n_agents, 1)
        dist = self.distance(p1, p2).squeeze(4)
        ineqs_inputs = {'pos_z': states[:, :, 1:, 2], 'speed': torch.norm(
            velocity[:, :, 1:, :], p=2, dim=3), 'acc': torch.norm(acc[:, :,
            1:, :], p=2, dim=3), 'dist_to_ball': dist[:, :, 0, 1:]}
        output = torch.cat([self.ineqs[k](ineqs_inputs[k], beta, **kwargs) for
            k in self.primitive_list], dim=-1)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
