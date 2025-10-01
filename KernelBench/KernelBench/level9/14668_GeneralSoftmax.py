import enum
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def _canonize_enum_value(value):
    if type(value) is str:
        value = value.lower()
    return value


def masked_softmax(logits, mask=None, dim=-1):
    eps = 1e-20
    probs = F.softmax(logits, dim=dim)
    if mask is not None:
        mask = mask.float()
        probs = probs * mask + eps
        probs = probs / probs.sum(dim, keepdim=True)
    return probs


def no_grad_func(func):

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func


@no_grad_func
def one_hot(index, nr_classes):
    """
    Convert a list of class labels into one-hot representation.

    .. note::
        This function support only one-dimensional input. For high dimensional inputs, use `one_hot_nd`.

    Args:
        index (Tensor): shape `(N, )`, input class labels.
        nr_classes (int): number of total classes.

    Returns:
        Tensor: shape `(N, nr_classes)`, one-hot representation of the class labels.

    """
    assert index.dim() == 1
    mask = torch.zeros(index.size(0), nr_classes, dtype=torch.float32,
        device=index.device)
    ones = torch.ones(index.size(0), 1, dtype=torch.float32, device=index.
        device)
    ret = mask.scatter_(1, index.unsqueeze(1), ones)
    return ret


@no_grad_func
def one_hot_nd(index, nr_classes):
    """
    Convert a tensor of class labels into one-hot representation.

    Args:
        index (Tensor): input class labels.
        nr_classes (int): number of total classes.

    Returns:
        Tensor: one-hot representation of the class labels, the label dimension is assumed to be the last one.

    """
    index_size = index.size()
    return one_hot(index.view(-1), nr_classes).view(index_size + (nr_classes,))


def greedy_softmax(logits, dim=-1, mask=None):
    assert dim == -1, 'Greedy softmax support only dim=-1'
    if mask is not None:
        probs = masked_softmax(logits, mask=mask, dim=dim)
    else:
        probs = logits
    one_hot = one_hot_nd(probs.max(dim)[1], logits.size(dim))
    return one_hot


def _sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return -torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, dim=-1, tau=1, eps=1e-10, mask=None):
    """
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps, out=logits.new())
    y = logits + gumbel_noise
    return masked_softmax(y / tau, mask, dim=dim)


def set_index_one_hot_(tensor, dim, index, value):
    """
    `tensor[:, :, index, :, :] = value`.

    Args:
        tensor (Tensor): input.
        dim (int) the dimension.
        index: (LongTensor): the tensor containing the indices along the `dim` dimension.

    """
    if not isinstance(value, (int, float)):
        value = value.unsqueeze(dim)
    tensor.scatter_(dim, index.unsqueeze(dim), value)


def gumbel_softmax(logits, dim=-1, tau=1, hard=False, mask=None, eps=1e-10):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        dim: along which dim the softmax is performed
        tau: non-negative scalar temperature
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        eps: eps

    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probability distribution that sums to 1 across classes

    Based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = _gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        with torch.no_grad():
            _, k = y_soft.max(dim=dim)
            y_hard = torch.zeros_like(logits)
            y_hard.requires_grad = False
            set_index_one_hot_(y_hard, dim, k, 1.0)
        y = (y_hard - y_soft).detach() + y_soft
    else:
        y = y_soft
    return y


def general_softmax(logits, dim=-1, tau=1, impl='standard', mask=None,
    training=False):
    impl = SoftmaxImplmentation.from_string(impl)
    if impl is SoftmaxImplmentation.STANDARD:
        return masked_softmax(logits / tau, dim=dim)
    elif impl in (SoftmaxImplmentation.GUMBEL, SoftmaxImplmentation.GUMBEL_HARD
        ):
        if not training:
            return greedy_softmax(logits, dim=dim, mask=mask)
        if impl is SoftmaxImplmentation.GUMBEL:
            return gumbel_softmax(logits, dim=dim, tau=tau, hard=False,
                mask=mask)
        else:
            return gumbel_softmax(logits, dim=dim, tau=tau, hard=True, mask
                =mask)


class JacEnum(enum.Enum):
    """A customized enumeration class, adding helper functions for string-based argument parsing."""

    @classmethod
    def from_string(cls, value):
        value = _canonize_enum_value(value)
        return cls(value)

    @classmethod
    def type_name(cls):
        return cls.__name__

    @classmethod
    def choice_names(cls):
        return list(filter(lambda x: not x.startswith('_'), dir(cls)))

    @classmethod
    def choice_objs(cls):
        return [getattr(cls, name) for name in cls.choice_names()]

    @classmethod
    def choice_values(cls):
        return [getattr(cls, name).value for name in cls.choice_names()]

    @classmethod
    def is_valid(cls, value):
        value = _canonize_enum_value(value)
        return value in cls.choice_values()

    @classmethod
    def assert_valid(cls, value):
        assert cls.is_valid(value
            ), 'Invalid {}: "{}". Supported choices: {}.'.format(cls.
            type_name(), value, ','.join(cls.choice_values()))

    def __jsonify__(self):
        return self.value


class SoftmaxImplmentation(JacEnum):
    STANDARD = 'standard'
    GUMBEL = 'gumbel'
    GUMBEL_HARD = 'gumbel_hard'


class GeneralSoftmax(nn.Module):

    def __init__(self, dim=-1, tau=1.0, impl='standard'):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.impl = SoftmaxImplmentation.from_string(impl)

    def forward(self, logits, mask=None):
        return general_softmax(logits, dim=self.dim, tau=self.tau, impl=
            self.impl, mask=mask, training=self.training)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
