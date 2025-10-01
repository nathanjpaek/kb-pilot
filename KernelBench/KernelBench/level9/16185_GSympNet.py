from torch.nn import Module
import torch
import torch.nn as nn


class Module(torch.nn.Module):
    """Standard module format. 
    """

    def __init__(self):
        super(Module, self).__init__()
        self.activation = None
        self.initializer = None
        self.__device = None
        self.__dtype = None

    @property
    def device(self):
        return self.__device

    @property
    def dtype(self):
        return self.__dtype

    @device.setter
    def device(self, d):
        if d == 'cpu':
            self.cpu()
        elif d == 'gpu':
            self
        else:
            raise ValueError
        self.__device = d

    @dtype.setter
    def dtype(self, d):
        if d == 'float':
            self
        elif d == 'double':
            self
        else:
            raise ValueError
        self.__dtype = d

    @property
    def Device(self):
        if self.__device == 'cpu':
            return torch.device('cpu')
        elif self.__device == 'gpu':
            return torch.device('cuda')

    @property
    def Dtype(self):
        if self.__dtype == 'float':
            return torch.float32
        elif self.__dtype == 'double':
            return torch.float64

    @property
    def act(self):
        if self.activation == 'sigmoid':
            return torch.sigmoid
        elif self.activation == 'relu':
            return torch.relu
        elif self.activation == 'tanh':
            return torch.tanh
        elif self.activation == 'elu':
            return torch.elu
        else:
            raise NotImplementedError

    @property
    def Act(self):
        if self.activation == 'sigmoid':
            return torch.nn.Sigmoid()
        elif self.activation == 'relu':
            return torch.nn.ReLU()
        elif self.activation == 'tanh':
            return torch.nn.Tanh()
        elif self.activation == 'elu':
            return torch.nn.ELU()
        else:
            raise NotImplementedError

    @property
    def weight_init_(self):
        if self.initializer == 'He normal':
            return torch.nn.init.kaiming_normal_
        elif self.initializer == 'He uniform':
            return torch.nn.init.kaiming_uniform_
        elif self.initializer == 'Glorot normal':
            return torch.nn.init.xavier_normal_
        elif self.initializer == 'Glorot uniform':
            return torch.nn.init.xavier_uniform_
        elif self.initializer == 'orthogonal':
            return torch.nn.init.orthogonal_
        elif self.initializer == 'default':
            if self.activation == 'relu':
                return torch.nn.init.kaiming_normal_
            elif self.activation == 'tanh':
                return torch.nn.init.orthogonal_
            else:
                return lambda x: None
        else:
            raise NotImplementedError


class StructureNN(Module):
    """Structure-oriented neural network used as a general map based on designing architecture.
    """

    def __init__(self):
        super(StructureNN, self).__init__()

    def predict(self, x, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.Dtype, device=self.Device)
        return self(x).cpu().detach().numpy() if returnnp else self(x)


class GradientModule(Module):
    """Gradient symplectic module.
    """

    def __init__(self, dim, width, activation, mode):
        super(GradientModule, self).__init__()
        self.dim = dim
        self.width = width
        self.activation = activation
        self.mode = mode
        self.params = self.__init_params()

    def forward(self, pqh):
        p, q, h = pqh
        if self.mode == 'up':
            gradH = self.act(q @ self.params['K'] + self.params['b']
                ) * self.params['a'] @ self.params['K'].t()
            return p + gradH * h, q
        elif self.mode == 'low':
            gradH = self.act(p @ self.params['K'] + self.params['b']
                ) * self.params['a'] @ self.params['K'].t()
            return p, gradH * h + q
        else:
            raise ValueError

    def __init_params(self):
        d = int(self.dim / 2)
        params = nn.ParameterDict()
        params['K'] = nn.Parameter((torch.randn([d, self.width]) * 0.01).
            requires_grad_(True))
        params['a'] = nn.Parameter((torch.randn([self.width]) * 0.01).
            requires_grad_(True))
        params['b'] = nn.Parameter(torch.zeros([self.width]).requires_grad_
            (True))
        return params


class SympNet(StructureNN):

    def __init__(self):
        super(SympNet, self).__init__()
        self.dim = None

    def predict(self, xh, steps=1, keepinitx=False, returnnp=False):
        dim = xh.size(-1)
        size = len(xh.size())
        if dim == self.dim:
            pred = [xh]
            for _ in range(steps):
                pred.append(self(pred[-1]))
        else:
            x0, h = xh[..., :-1], xh[..., -1:]
            pred = [x0]
            for _ in range(steps):
                pred.append(self(torch.cat([pred[-1], h], dim=-1)))
        if keepinitx:
            steps = steps + 1
        else:
            pred = pred[1:]
        res = torch.cat(pred, dim=-1).view([-1, steps, self.dim][2 - size:])
        return res.cpu().detach().numpy() if returnnp else res


class GSympNet(SympNet):
    """G-SympNet.
    Input: [num, dim] or [num, dim + 1]
    Output: [num, dim]
    """

    def __init__(self, dim, layers=3, width=20, activation='sigmoid'):
        super(GSympNet, self).__init__()
        self.dim = dim
        self.layers = layers
        self.width = width
        self.activation = activation
        self.modus = self.__init_modules()

    def forward(self, pqh):
        d = int(self.dim / 2)
        if pqh.size(-1) == self.dim + 1:
            p, q, h = pqh[..., :d], pqh[..., d:-1], pqh[..., -1:]
        elif pqh.size(-1) == self.dim:
            p, q, h = pqh[..., :d], pqh[..., d:], torch.ones_like(pqh[..., -1:]
                )
        else:
            raise ValueError
        for i in range(self.layers):
            GradM = self.modus['GradM{}'.format(i + 1)]
            p, q = GradM([p, q, h])
        return torch.cat([p, q], dim=-1)

    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(self.layers):
            mode = 'up' if i % 2 == 0 else 'low'
            modules['GradM{}'.format(i + 1)] = GradientModule(self.dim,
                self.width, self.activation, mode)
        return modules


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
