from torch.autograd import Function
import torch
import numpy as np
from warnings import warn
from typing import Union
from typing import Tuple
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single
import torch.nn.functional as F


def _cayley(A):
    I = torch.eye(A.shape[-1], device=A.device)
    LU = torch.lu(I + A, pivot=True)
    return torch.lu_solve(I - A, *LU)


def _cayley_frechet(A, H, Q=None):
    I = torch.eye(A.shape[-1], device=A.device)
    if Q is None:
        Q = _cayley(A)
    _LU = torch.lu(I + A, pivot=True)
    p = torch.lu_solve(Q, *_LU)
    _LU = torch.lu(I - A, pivot=True)
    q = torch.lu_solve(H, *_LU)
    return 2.0 * q @ p


def __calculate_kernel_matrix_cayley__(weight, **kwargs):
    skew_symmetric_matrix = weight - torch.transpose(weight, -1, -2)
    return cayley.apply(skew_symmetric_matrix)


def matrix_1_norm(A):
    """Calculates the 1-norm of a matrix or a batch of matrices.

    Args:
        A (torch.Tensor): Can be either of size (n,n) or (m,n,n).

    Returns:
        torch.Tensor : The 1-norm of A.
    """
    norm, _indices = torch.max(torch.sum(torch.abs(A), axis=-2), axis=-1)
    return norm


def _compute_scales(A):
    """Compute optimal parameters for scaling-and-squaring algorithm.

    The constants used in this function are determined by the MATLAB
    function found in
    https://github.com/cetmann/pytorch_expm/blob/master/determine_frechet_scaling_constant.m
    """
    norm = matrix_1_norm(A)
    max_norm = torch.max(norm)
    s = torch.zeros_like(norm)
    if A.dtype == torch.float64:
        if A.requires_grad:
            ell = {(3): 0.010813385777848, (5): 0.199806320697895, (7): 
                0.783460847296204, (9): 1.782448623969279, (13): 
                4.740307543765127}
        else:
            ell = {(3): 0.014955852179582, (5): 0.253939833006323, (7): 
                0.950417899616293, (9): 2.097847961257068, (13): 
                5.371920351148152}
        if max_norm >= ell[9]:
            m = 13
            magic_number = ell[m]
            s = torch.relu_(torch.ceil(torch.log2_(norm / magic_number)))
        else:
            for m in [3, 5, 7, 9]:
                if max_norm < ell[m]:
                    magic_number = ell[m]
                    break
    elif A.dtype == torch.float32:
        if A.requires_grad:
            ell = {(3): 0.30803304184533, (5): 1.482532614793145, (7): 
                3.248671755200478}
        else:
            ell = {(3): 0.425873001692283, (5): 1.880152677804762, (7): 
                3.92572478313866}
        if max_norm >= ell[5]:
            m = 7
            magic_number = ell[m]
            s = torch.relu_(torch.ceil(torch.log2_(norm / magic_number)))
        else:
            for m in [3, 5]:
                if max_norm < ell[m]:
                    magic_number = ell[m]
                    break
    return s, m


def _eye_like(M, device=None, dtype=None):
    """Creates an identity matrix of the same shape as another matrix.

    For matrix M, the output is same shape as M, if M is a (n,n)-matrix.
    If M is a batch of m matrices (i.e. a (m,n,n)-tensor), create a batch of
    (n,n)-identity-matrices.

    Args:
        M (torch.Tensor) : A tensor of either shape (n,n) or (m,n,n), for
            which either an identity matrix or a batch of identity matrices
            of the same shape will be created.
        device (torch.device, optional) : The device on which the output
            will be placed. By default, it is placed on the same device
            as M.
        dtype (torch.dtype, optional) : The dtype of the output. By default,
            it is the same dtype as M.

    Returns:
        torch.Tensor : Identity matrix or batch of identity matrices.
    """
    assert len(M.shape) in [2, 3]
    assert M.shape[-1] == M.shape[-2]
    n = M.shape[-1]
    if device is None:
        device = M.device
    if dtype is None:
        dtype = M.dtype
    eye = torch.eye(M.shape[-1], device=device, dtype=dtype)
    if len(M.shape) == 2:
        return eye
    else:
        m = M.shape[0]
        return eye.view(-1, n, n).expand(m, -1, -1)


def _expm_frechet_pade(A, E, m=7):
    assert m in [3, 5, 7, 9, 13]
    if m == 3:
        b = [120.0, 60.0, 12.0, 1.0]
    elif m == 5:
        b = [30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0]
    elif m == 7:
        b = [17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 
            56.0, 1.0]
    elif m == 9:
        b = [17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 
            30270240.0, 2162160.0, 110880.0, 3960.0, 90.0, 1.0]
    elif m == 13:
        b = [6.476475253248e+16, 3.238237626624e+16, 7771770303897600.0, 
            1187353796428800.0, 129060195264000.0, 10559470521600.0, 
            670442572800.0, 33522128640.0, 1323241920.0, 40840800.0, 
            960960.0, 16380.0, 182.0, 1.0]
    I = _eye_like(A)
    if m != 13:
        if m >= 3:
            M_2 = A @ E + E @ A
            A_2 = A @ A
            U = b[3] * A_2
            V = b[2] * A_2
            L_U = b[3] * M_2
            L_V = b[2] * M_2
        if m >= 5:
            M_4 = A_2 @ M_2 + M_2 @ A_2
            A_4 = A_2 @ A_2
            U = U + b[5] * A_4
            V = V + b[4] * A_4
            L_U = L_U + b[5] * M_4
            L_V = L_V + b[4] * M_4
        if m >= 7:
            M_6 = A_4 @ M_2 + M_4 @ A_2
            A_6 = A_4 @ A_2
            U = U + b[7] * A_6
            V = V + b[6] * A_6
            L_U = L_U + b[7] * M_6
            L_V = L_V + b[6] * M_6
        if m == 9:
            M_8 = A_4 @ M_4 + M_4 @ A_4
            A_8 = A_4 @ A_4
            U = U + b[9] * A_8
            V = V + b[8] * A_8
            L_U = L_U + b[9] * M_8
            L_V = L_V + b[8] * M_8
        U = U + b[1] * I
        V = U + b[0] * I
        del I
        L_U = A @ L_U
        L_U = L_U + E @ U
        U = A @ U
    else:
        M_2 = A @ E + E @ A
        A_2 = A @ A
        M_4 = A_2 @ M_2 + M_2 @ A_2
        A_4 = A_2 @ A_2
        M_6 = A_4 @ M_2 + M_4 @ A_2
        A_6 = A_4 @ A_2
        W_1 = b[13] * A_6 + b[11] * A_4 + b[9] * A_2
        W_2 = b[7] * A_6 + b[5] * A_4 + b[3] * A_2 + b[1] * I
        W = A_6 @ W_1 + W_2
        Z_1 = b[12] * A_6 + b[10] * A_4 + b[8] * A_2
        Z_2 = b[6] * A_6 + b[4] * A_4 + b[2] * A_2 + b[0] * I
        U = A @ W
        V = A_6 @ Z_1 + Z_2
        L_W1 = b[13] * M_6 + b[11] * M_4 + b[9] * M_2
        L_W2 = b[7] * M_6 + b[5] * M_4 + b[3] * M_2
        L_Z1 = b[12] * M_6 + b[10] * M_4 + b[8] * M_2
        L_Z2 = b[6] * M_6 + b[4] * M_4 + b[2] * M_2
        L_W = A_6 @ L_W1 + M_6 @ W_1 + L_W2
        L_U = A @ L_W + E @ W
        L_V = A_6 @ L_Z1 + M_6 @ Z_1 + L_Z2
    lu_decom = torch.lu(-U + V)
    exp_A = torch.lu_solve(U + V, *lu_decom)
    dexp_A = torch.lu_solve(L_U + L_V + (L_U - L_V) @ exp_A, *lu_decom)
    return exp_A, dexp_A


def _square(s, R, L=None):
    """The `squaring` part of the `scaling-and-squaring` algorithm.

    This works both for the forward as well as the derivative of
    the matrix exponential.
    """
    s_max = torch.max(s).int()
    if s_max > 0:
        I = _eye_like(R)
        if L is not None:
            O = torch.zeros_like(R)
        indices = [(1) for k in range(len(R.shape) - 1)]
    for i in range(s_max):
        mask = i >= s
        matrices_mask = mask.view(-1, *indices)
        temp_eye = torch.clone(R).masked_scatter(matrices_mask, I)
        if L is not None:
            temp_zeros = torch.clone(R).masked_scatter(matrices_mask, O)
            L = temp_eye @ L + temp_zeros @ L
        R = R @ temp_eye
        del temp_eye, mask
    if L is not None:
        return R, L
    else:
        return R


def _expm_frechet_scaling_squaring(A, E, adjoint=False):
    """Numerical Fréchet derivative of matrix exponentiation.

    """
    assert A.shape[-1] == A.shape[-2] and len(A.shape) in [2, 3]
    True if len(A.shape) == 3 else False
    if adjoint is True:
        A = torch.transpose(A, -1, -2)
    s, m = _compute_scales(A)
    if torch.max(s) > 0:
        indices = [(1) for k in range(len(A.shape) - 1)]
        scaling_factors = torch.pow(2, -s).view(-1, *indices)
        A = A * scaling_factors
        E = E * scaling_factors
    exp_A, dexp_A = _expm_frechet_pade(A, E, m)
    exp_A, dexp_A = _square(s, exp_A, dexp_A)
    return dexp_A


def _expm_pade(A, m=7):
    assert m in [3, 5, 7, 9, 13]
    if m == 3:
        b = [120.0, 60.0, 12.0, 1.0]
    elif m == 5:
        b = [30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0]
    elif m == 7:
        b = [17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 
            56.0, 1.0]
    elif m == 9:
        b = [17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 
            30270240.0, 2162160.0, 110880.0, 3960.0, 90.0, 1.0]
    elif m == 13:
        b = [6.476475253248e+16, 3.238237626624e+16, 7771770303897600.0, 
            1187353796428800.0, 129060195264000.0, 10559470521600.0, 
            670442572800.0, 33522128640.0, 1323241920.0, 40840800.0, 
            960960.0, 16380.0, 182.0, 1.0]
    I = _eye_like(A)
    if m != 13:
        U = b[1] * I
        V = b[0] * I
        if m >= 3:
            A_2 = A @ A
            U = U + b[3] * A_2
            V = V + b[2] * A_2
        if m >= 5:
            A_4 = A_2 @ A_2
            U = U + b[5] * A_4
            V = V + b[4] * A_4
        if m >= 7:
            A_6 = A_4 @ A_2
            U = U + b[7] * A_6
            V = V + b[6] * A_6
        if m == 9:
            A_8 = A_4 @ A_4
            U = U + b[9] * A_8
            V = V + b[8] * A_8
        U = A @ U
    else:
        A_2 = A @ A
        A_4 = A_2 @ A_2
        A_6 = A_4 @ A_2
        W_1 = b[13] * A_6 + b[11] * A_4 + b[9] * A_2
        W_2 = b[7] * A_6 + b[5] * A_4 + b[3] * A_2 + b[1] * I
        W = A_6 @ W_1 + W_2
        Z_1 = b[12] * A_6 + b[10] * A_4 + b[8] * A_2
        Z_2 = b[6] * A_6 + b[4] * A_4 + b[2] * A_2 + b[0] * I
        U = A @ W
        V = A_6 @ Z_1 + Z_2
    del A_2
    if m >= 5:
        del A_4
    if m >= 7:
        del A_6
    if m == 9:
        del A_8
    R = torch.lu_solve(U + V, *torch.lu(-U + V))
    del U, V
    return R


def _expm_scaling_squaring(A):
    """Scaling-and-squaring algorithm for matrix eponentiation.

    This is based on the observation that exp(A) = exp(A/k)^k, where
    e.g. k=2^s. The exponential exp(A/(2^s)) is calculated by a diagonal
    Padé approximation, where s is chosen based on the 1-norm of A, such
    that certain approximation guarantees can be given. exp(A) is then
    calculated by repeated squaring via exp(A/(2^s))^(2^s). This function
    works both for (n,n)-tensors as well as batchwise for (m,n,n)-tensors.
    """
    assert A.shape[-1] == A.shape[-2] and len(A.shape) in [2, 3]
    True if len(A.shape) == 3 else False
    s, m = _compute_scales(A)
    if torch.max(s) > 0:
        indices = [(1) for k in range(len(A.shape) - 1)]
        A = A * torch.pow(2, -s).view(-1, *indices)
    exp_A = _expm_pade(A, m)
    exp_A = _square(s, exp_A)
    return exp_A


def __calculate_kernel_matrix_exp__(weight, **kwargs):
    skew_symmetric_matrix = weight - torch.transpose(weight, -1, -2)
    return expm.apply(skew_symmetric_matrix)


def eye_like(M, device=None, dtype=None):
    """Creates an identity matrix of the same shape as another matrix.

    For matrix M, the output is same shape as M, if M is a (n,n)-matrix.
    If M is a batch of m matrices (i.e. a (m,n,n)-tensor), create a batch of
    (n,n)-identity-matrices.

    Args:
        M (torch.Tensor) : A tensor of either shape (n,n) or (m,n,n), for
            which either an identity matrix or a batch of identity matrices
            of the same shape will be created.
        device (torch.device, optional) : The device on which the output
            will be placed. By default, it is placed on the same device
            as M.
        dtype (torch.dtype, optional) : The dtype of the output. By default,
            it is the same dtype as M.

    Returns:
        torch.Tensor : Identity matrix or batch of identity matrices.
    """
    assert len(M.shape) in [2, 3]
    assert M.shape[-1] == M.shape[-2]
    n = M.shape[-1]
    if device is None:
        device = M.device
    if dtype is None:
        dtype = M.dtype
    eye = torch.eye(M.shape[-1], device=device, dtype=dtype)
    if len(M.shape) == 2:
        return eye
    else:
        m = M.shape[0]
        return eye.view(-1, n, n).expand(m, -1, -1)


def householder_matrix(unit_vector):
    if unit_vector.shape[-1] != 1:
        if len(unit_vector.shape) == 1:
            return torch.ones_like(unit_vector)
        unit_vector = unit_vector.view(*tuple(unit_vector.shape), 1)
    transform = 2 * unit_vector @ torch.transpose(unit_vector, -1, -2)
    return eye_like(transform) - transform


def normalize_matrix_rows(matrix, eps=1e-06):
    norms = torch.sqrt(torch.sum(matrix ** 2, dim=-2, keepdim=True) + eps)
    return matrix / norms


def householder_transform(matrix, n_reflections=-1, eps=1e-06):
    """Implements a product of Householder transforms.

    """
    if n_reflections == -1:
        n_reflections = matrix.shape[-1]
    if n_reflections > matrix.shape[-1]:
        warn('n_reflections is set higher than the number of rows.')
        n_reflections = matrix.shape[-1]
    matrix = normalize_matrix_rows(matrix, eps)
    if n_reflections == 0:
        output = torch.eye(matrix.shape[-2], dtype=matrix.dtype, device=
            matrix.device)
        if len(matrix.shape) == 3:
            output = output.view(1, matrix.shape[1], matrix.shape[1])
            output = output.expand(matrix.shape[0], -1, -1)
    for i in range(n_reflections):
        unit_vector = matrix[..., i:i + 1]
        householder = householder_matrix(unit_vector)
        if i == 0:
            output = householder
        else:
            output = output @ householder
    return output


def __calculate_kernel_matrix_householder__(weight, **kwargs):
    n_reflections = kwargs.get('n_reflections', -1)
    eps = kwargs.get('eps', 1e-06)
    weight.shape[-1]
    weight = weight[..., n_reflections:]
    return householder_transform(weight, n_reflections, eps)


def __initialize_weight__(kernel_matrix_shape: 'Tuple[int, ...]', stride:
    'Tuple[int, ...]', method: 'str'='cayley', init: 'str'='haar', dtype:
    'str'='float32', *args, **kwargs):
    """Function which computes specific orthogonal matrices.

    For some chosen method of parametrizing orthogonal matrices, this
    function outputs the required weights necessary to represent a
    chosen initialization as a Pytorch tensor of matrices.

    Args:
        kernel_matrix_shape : The output shape of the
            orthogonal matrices. Should be (num_matrices, height, width).
        stride : The stride for the invertible up- or
            downsampling for which this matrix is to be used. The length
            of ``stride`` should match the dimensionality of the data.
        method : The method for parametrising orthogonal matrices.
            Should be 'exp' or 'cayley'
        init : The matrix which should be represented. Should be
            'squeeze', 'pixel_shuffle', 'haar' or 'random'. 'haar' is only
            possible if ``stride`` is only 2.
        dtype : Numpy dtype which should be used for the matrix.
        *args: Variable length argument iterable.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor : Orthogonal matrices of shape ``kernel_matrix_shape``.
    """
    dim = len(stride)
    num_matrices = kernel_matrix_shape[0]
    assert method in ['exp', 'cayley', 'householder']
    if method == 'householder':
        warn(
            'Householder parametrization not fully implemented yet. Only random initialization currently working.'
            )
        init = 'random'
    if init == 'random':
        return torch.randn(kernel_matrix_shape)
    if init == 'haar' and set(stride) != {2}:
        None
        None
        init = 'squeeze'
    if init == 'haar' and set(stride) == {2}:
        if method == 'exp':
            p = np.pi / 4
            if dim == 1:
                weight = np.array([[[0, p], [0, 0]]], dtype=dtype)
            if dim == 2:
                weight = np.array([[[0, 0, p, p], [0, 0, -p, -p], [0, 0, 0,
                    0], [0, 0, 0, 0]]], dtype=dtype)
            if dim == 3:
                weight = np.array([[[0, p, p, 0, p, 0, 0, 0], [0, 0, 0, p, 
                    0, p, 0, 0], [0, 0, 0, p, 0, 0, p, 0], [0, 0, 0, 0, 0, 
                    0, 0, p], [0, 0, 0, 0, 0, p, p, 0], [0, 0, 0, 0, 0, 0, 
                    0, p], [0, 0, 0, 0, 0, 0, 0, p], [0, 0, 0, 0, 0, 0, 0, 
                    0]]], dtype=dtype)
            return torch.tensor(weight).repeat(num_matrices, 1, 1)
        elif method == 'cayley':
            if dim == 1:
                p = -np.sqrt(2) / (2 - np.sqrt(2))
                weight = np.array([[[0, p], [0, 0]]], dtype=dtype)
            if dim == 2:
                p = 0.5
                weight = np.array([[[0, 0, p, p], [0, 0, -p, -p], [0, 0, 0,
                    0], [0, 0, 0, 0]]], dtype=dtype)
            if dim == 3:
                p = 1 / np.sqrt(2)
                weight = np.array([[[0, -p, -p, 0, -p, 0, 0, 1 - p], [0, 0,
                    0, -p, 0, -p, p - 1, 0], [0, 0, 0, -p, 0, p - 1, -p, 0],
                    [0, 0, 0, 0, 1 - p, 0, 0, -p], [0, 0, 0, 0, 0, -p, -p, 
                    0], [0, 0, 0, 0, 0, 0, 0, -p], [0, 0, 0, 0, 0, 0, 0, -p
                    ], [0, 0, 0, 0, 0, 0, 0, 0]]], dtype=dtype)
            return torch.tensor(weight).repeat(num_matrices, 1, 1)
    if init in ['squeeze', 'pixel_shuffle', 'zeros']:
        if method == 'exp' or method == 'cayley':
            return torch.zeros(*kernel_matrix_shape)
    if type(init) is np.ndarray:
        init = torch.tensor(init.astype(dtype))
    if torch.is_tensor(init):
        if len(init.shape) == 2:
            init = init.reshape(1, *init.shape)
        if init.shape[0] == 1:
            init = init.repeat(num_matrices, 1, 1)
        assert init.shape == kernel_matrix_shape
        return init
    else:
        raise NotImplementedError('Unknown initialization.')


class cayley(Function):
    """Computes the Cayley transform.

    """

    @staticmethod
    def forward(ctx, M):
        cayley_M = _cayley(M)
        ctx.save_for_backward(M, cayley_M)
        return cayley_M

    @staticmethod
    def backward(ctx, grad_out):
        M, cayley_M = ctx.saved_tensors
        dcayley_M = _cayley_frechet(M, grad_out, Q=cayley_M)
        return dcayley_M


class expm(Function):
    """Computes the matrix exponential.

    """

    @staticmethod
    def forward(ctx, M):
        expm_M = _expm_scaling_squaring(M)
        ctx.save_for_backward(M)
        return expm_M

    @staticmethod
    def backward(ctx, grad_out):
        M = ctx.saved_tensors[0]
        dexpm = _expm_frechet_scaling_squaring(M, grad_out, adjoint=True)
        return dexpm


class OrthogonalResamplingLayer(torch.nn.Module):
    """Base class for orthogonal up- and downsampling operators.

    :param low_channel_number:
        Lower number of channels. These are the input
        channels in the case of downsampling ops, and the output
        channels in the case of upsampling ops.
    :param stride:
        The downsampling / upsampling factor for each dimension.
    :param channel_multiplier:
        The channel multiplier, i.e. the number
        by which the number of channels are multiplied (downsampling)
        or divided (upsampling).
    :param method:
        Which method to use for parametrizing orthogonal
        matrices which are used as convolutional kernels.
    """

    def __init__(self, low_channel_number: 'int', stride:
        'Union[int, Tuple[int, ...]]', method: 'str'='cayley', init:
        'Union[str, np.ndarray, torch.Tensor]'='haar', learnable: 'bool'=
        True, init_kwargs: 'dict'=None, **kwargs):
        super(OrthogonalResamplingLayer, self).__init__()
        self.low_channel_number = low_channel_number
        self.method = method
        self.stride = stride
        self.channel_multiplier = int(np.prod(stride))
        self.high_channel_number = self.channel_multiplier * low_channel_number
        if init_kwargs is None:
            init_kwargs = {}
        self.init_kwargs = init_kwargs
        self.kwargs = kwargs
        assert method in ['exp', 'cayley', 'householder']
        if method == 'exp':
            self.__calculate_kernel_matrix__ = __calculate_kernel_matrix_exp__
        elif method == 'cayley':
            self.__calculate_kernel_matrix__ = (
                __calculate_kernel_matrix_cayley__)
        elif method == 'householder':
            self.__calculate_kernel_matrix__ = (
                __calculate_kernel_matrix_householder__)
        self._kernel_matrix_shape = (self.low_channel_number,) + (self.
            channel_multiplier,) * 2
        self._kernel_shape = (self.high_channel_number, 1) + self.stride
        self.weight = torch.nn.Parameter(__initialize_weight__(
            kernel_matrix_shape=self._kernel_matrix_shape, stride=self.
            stride, method=self.method, init=init, **self.init_kwargs))
        self.weight.requires_grad = learnable

    @property
    def kernel_matrix(self):
        """The orthogonal matrix created by the chosen parametrisation method.
        """
        return self.__calculate_kernel_matrix__(self.weight, **self.kwargs)

    @property
    def kernel(self):
        """The kernel associated with the invertible up-/downsampling.
        """
        return self.kernel_matrix.reshape(*self._kernel_shape)


class InvertibleDownsampling1D(OrthogonalResamplingLayer):

    def __init__(self, in_channels: 'int', stride: '_size_1_t'=2, method:
        'str'='cayley', init: 'str'='haar', learnable: 'bool'=True, *args,
        **kwargs):
        stride = tuple(_single(stride))
        channel_multiplier = int(np.prod(stride))
        self.in_channels = in_channels
        self.out_channels = in_channels * channel_multiplier
        super(InvertibleDownsampling1D, self).__init__(*args,
            low_channel_number=self.in_channels, stride=stride, method=
            method, init=init, learnable=learnable, **kwargs)

    def forward(self, x):
        return F.conv1d(x, self.kernel, stride=self.stride, groups=self.
            low_channel_number)

    def inverse(self, x):
        return F.conv_transpose1d(x, self.kernel, stride=self.stride,
            groups=self.low_channel_number)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
