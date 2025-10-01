import torch
from torch import nn


def _extra_repr(**kwargs):
    if 'n' in kwargs:
        ret = 'n={}'.format(kwargs['n'])
    elif 'dim' in kwargs:
        ret = 'dim={}'.format(kwargs['dim'])
    else:
        ret = ''
    if 'k' in kwargs:
        ret += ', k={}'.format(kwargs['k'])
    if 'rank' in kwargs:
        ret += ', rank={}'.format(kwargs['rank'])
    if 'radius' in kwargs:
        ret += ', radius={}'.format(kwargs['radius'])
    if 'lam' in kwargs:
        ret += ', lambda={}'.format(kwargs['lam'])
    if 'f' in kwargs:
        ret += ', f={}'.format(kwargs['f'].__name__)
    if 'tensorial_size' in kwargs:
        ts = kwargs['tensorial_size']
        if len(ts) != 0:
            ret += ', tensorial_size={}'.format(tuple(ts))
    if 'triv' in kwargs:
        ret += ', triv={}'.format(kwargs['triv'].__name__)
    if 'no_inv' in kwargs:
        if kwargs['no_inv']:
            ret += ', no inverse'
    if 'transposed' in kwargs:
        if kwargs['transposed']:
            ret += ', transposed'
    return ret


def _in_sphere(x, r, eps):
    norm = x.norm(dim=-1)
    rs = torch.full_like(norm, r)
    return (torch.norm(norm - rs, p=float('inf')) < eps).all()


def project(x):
    return x / x.norm(dim=-1, keepdim=True)


def uniform_init_sphere_(x, r=1.0):
    """Samples a point uniformly on the sphere into the tensor ``x``.
    If ``x`` has :math:`d > 1` dimensions, the first :math:`d-1` dimensions
    are treated as batch dimensions.
    """
    with torch.no_grad():
        x.normal_()
        x.data = r * project(x)
    return x


class InManifoldError(ValueError):

    def __init__(self, X, M):
        super().__init__('Tensor not contained in {}. Got\n{}'.format(M, X))


class SphereEmbedded(nn.Module):

    def __init__(self, size, radius=1.0):
        """
        Sphere as the orthogonal projection from
        :math:`\\mathbb{R}^n` to :math:`\\mathbb{S}^{n-1}`, that is,
        :math:`x \\mapsto \\frac{x}{\\lVert x \\rVert}`.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            radius (float): Optional.
                Radius of the sphere. A positive number. Default: ``1.``
        """
        super().__init__()
        self.n = size[-1]
        self.tensorial_size = size[:-1]
        self.radius = SphereEmbedded.parse_radius(radius)

    @staticmethod
    def parse_radius(radius):
        if radius <= 0.0:
            raise ValueError(
                'The radius has to be a positive real number. Got {}'.
                format(radius))
        return radius

    def forward(self, x):
        return self.radius * project(x)

    def right_inverse(self, x, check_in_manifold=True):
        if check_in_manifold and not self.in_manifold(x):
            raise InManifoldError(x, self)
        return x / self.radius

    def in_manifold(self, x, eps=1e-05):
        """
        Checks that a vector is on the sphere.

        For tensors with more than 2 dimensions the first dimensions are
        treated as batch dimensions.

        Args:
            X (torch.Tensor): The vector to be checked.
            eps (float): Optional. Threshold at which the norm is considered
                    to be equal to ``1``. Default: ``1e-5``
        """
        return _in_sphere(x, self.radius, eps)

    def sample(self):
        """
        Returns a uniformly sampled vector on the sphere.
        """
        x = torch.empty(*(self.tensorial_size + (self.n,)))
        return uniform_init_sphere_(x, r=self.radius)

    def extra_repr(self):
        return _extra_repr(n=self.n, radius=self.radius, tensorial_size=
            self.tensorial_size)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': [4, 4]}]
