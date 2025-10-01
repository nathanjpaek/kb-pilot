from torch.nn import Module
import torch
import numpy as np
from numpy.random import RandomState
from torch.nn.parameter import Parameter


def quaternion_init(in_features, out_features, rng, kernel_size=None,
    criterion='glorot'):
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features
    if criterion == 'glorot':
        s = 1.0 / np.sqrt(2 * (fan_in + fan_out))
    elif criterion == 'he':
        s = 1.0 / np.sqrt(2 * fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)
    rng = RandomState(123)
    if kernel_size is None:
        kernel_shape = in_features, out_features
    elif type(kernel_size) is int:
        kernel_shape = (out_features, in_features) + tuple((kernel_size,))
    else:
        kernel_shape = (out_features, in_features) + (*kernel_size,)
    number_of_weights = np.prod(kernel_shape)
    v_i = np.random.normal(0.0, s, number_of_weights)
    v_j = np.random.normal(0.0, s, number_of_weights)
    v_k = np.random.normal(0.0, s, number_of_weights)
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
        v_i[i] /= norm
        v_j[i] /= norm
        v_k[i] /= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)
    modulus = rng.uniform(low=-s, high=s, size=kernel_shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
    weight_r = modulus * np.cos(phase)
    weight_i = modulus * v_i * np.sin(phase)
    weight_j = modulus * v_j * np.sin(phase)
    weight_k = modulus * v_k * np.sin(phase)
    return weight_r, weight_i, weight_j, weight_k


def unitary_init(in_features, out_features, rng, kernel_size=None,
    criterion='he'):
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features
    if criterion == 'glorot':
        s = 1.0 / np.sqrt(2 * (fan_in + fan_out))
    elif criterion == 'he':
        s = 1.0 / np.sqrt(2 * fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)
    if kernel_size is None:
        kernel_shape = in_features, out_features
    elif type(kernel_size) is int:
        kernel_shape = (out_features, in_features) + tuple((kernel_size,))
    else:
        kernel_shape = (out_features, in_features) + (*kernel_size,)
    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.normal(0.0, s, number_of_weights)
    v_i = np.random.normal(0.0, s, number_of_weights)
    v_j = np.random.normal(0.0, s, number_of_weights)
    v_k = np.random.normal(0.0, s, number_of_weights)
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_r[i] ** 2 + v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2
            ) + 0.0001
        v_r[i] /= norm
        v_i[i] /= norm
        v_j[i] /= norm
        v_k[i] /= norm
    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)
    return v_r, v_i, v_j, v_k


def affect_init(r_weight, i_weight, j_weight, k_weight, init_func, rng,
    init_criterion):
    if r_weight.size() != i_weight.size() or r_weight.size() != j_weight.size(
        ) or r_weight.size() != k_weight.size():
        raise ValueError(
            'The real and imaginary weights should have the same size. Found:'
             + ' r:' + str(r_weight.size()) + ' i:' + str(i_weight.size()) +
            ' j:' + str(j_weight.size()) + ' k:' + str(k_weight.size()))
    elif r_weight.dim() != 2:
        raise Exception(
            'affect_init accepts only matrices. Found dimension = ' + str(
            r_weight.dim()))
    kernel_size = None
    r, i, j, k = init_func(r_weight.size(0), r_weight.size(1), rng,
        kernel_size, init_criterion)
    r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j
        ), torch.from_numpy(k)
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)


def check_input(input):
    if input.dim() not in {2, 3}:
        raise RuntimeError(
            'quaternion linear accepts only input of dimension 2 or 3. input.dim = '
             + str(input.dim()))
    nb_hidden = input.size()[-1]
    if nb_hidden % 4 != 0:
        raise RuntimeError(
            'Quaternion Tensors must be divisible by 4. input.size()[1] = ' +
            str(nb_hidden))


def get_i(input):
    check_input(input)
    nb_hidden = input.size()[-1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden // 4, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden // 4, nb_hidden // 4)


def get_j(input):
    check_input(input)
    nb_hidden = input.size()[-1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden // 2, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden // 2, nb_hidden // 4)


def get_k(input):
    check_input(input)
    nb_hidden = input.size()[-1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden - nb_hidden // 4, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden - nb_hidden // 4, nb_hidden // 4)


def get_r(input):
    check_input(input)
    nb_hidden = input.size()[-1]
    if input.dim() == 2:
        return input.narrow(1, 0, nb_hidden // 4)
    elif input.dim() == 3:
        return input.narrow(2, 0, nb_hidden // 4)


class QuaternionLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, r_weight, i_weight, j_weight, k_weight, bias=None):
        ctx.save_for_backward(input, r_weight, i_weight, j_weight, k_weight,
            bias)
        check_input(input)
        cat_kernels_4_r = torch.cat((r_weight, -i_weight, -j_weight, -
            k_weight), dim=0)
        cat_kernels_4_i = torch.cat((i_weight, r_weight, -k_weight,
            j_weight), dim=0)
        cat_kernels_4_j = torch.cat((j_weight, k_weight, r_weight, -
            i_weight), dim=0)
        cat_kernels_4_k = torch.cat((k_weight, -j_weight, i_weight,
            r_weight), dim=0)
        cat_kernels_4_quaternion = torch.cat((cat_kernels_4_r,
            cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k), dim=1)
        if input.dim() == 2:
            if bias is not None:
                return torch.addmm(bias, input, cat_kernels_4_quaternion)
            else:
                return torch.mm(input, cat_kernels_4_quaternion)
        else:
            output = torch.matmul(input, cat_kernels_4_quaternion)
            if bias is not None:
                return output + bias
            else:
                return output

    @staticmethod
    def backward(ctx, grad_output):
        input, r_weight, i_weight, j_weight, k_weight, _bias = (ctx.
            saved_tensors)
        (grad_input) = (grad_weight_r) = (grad_weight_i) = (grad_weight_j) = (
            grad_weight_k) = (grad_bias) = None
        input_r = torch.cat((r_weight, -i_weight, -j_weight, -k_weight), dim=0)
        input_i = torch.cat((i_weight, r_weight, -k_weight, j_weight), dim=0)
        input_j = torch.cat((j_weight, k_weight, r_weight, -i_weight), dim=0)
        input_k = torch.cat((k_weight, -j_weight, i_weight, r_weight), dim=0)
        cat_kernels_4_quaternion_T = torch.cat((input_r, input_i, input_j,
            input_k), dim=1).permute(1, 0)
        cat_kernels_4_quaternion_T.requires_grad_(False)
        r = get_r(input)
        i = get_i(input)
        j = get_j(input)
        k = get_k(input)
        input_r = torch.cat((r, -i, -j, -k), dim=0)
        input_i = torch.cat((i, r, -k, j), dim=0)
        input_j = torch.cat((j, k, r, -i), dim=0)
        input_k = torch.cat((k, -j, i, r), dim=0)
        input_mat = torch.cat((input_r, input_i, input_j, input_k), dim=1)
        input_mat.requires_grad_(False)
        r = get_r(grad_output)
        i = get_i(grad_output)
        j = get_j(grad_output)
        k = get_k(grad_output)
        input_r = torch.cat((r, i, j, k), dim=1)
        input_i = torch.cat((-i, r, k, -j), dim=1)
        input_j = torch.cat((-j, -k, r, i), dim=1)
        input_k = torch.cat((-k, j, -i, r), dim=1)
        grad_mat = torch.cat((input_r, input_i, input_j, input_k), dim=0)
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(cat_kernels_4_quaternion_T)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_mat.permute(1, 0).mm(input_mat).permute(1, 0)
            unit_size_x = r_weight.size(0)
            unit_size_y = r_weight.size(1)
            grad_weight_r = grad_weight.narrow(0, 0, unit_size_x).narrow(1,
                0, unit_size_y)
            grad_weight_i = grad_weight.narrow(0, 0, unit_size_x).narrow(1,
                unit_size_y, unit_size_y)
            grad_weight_j = grad_weight.narrow(0, 0, unit_size_x).narrow(1,
                unit_size_y * 2, unit_size_y)
            grad_weight_k = grad_weight.narrow(0, 0, unit_size_x).narrow(1,
                unit_size_y * 3, unit_size_y)
        if ctx.needs_input_grad[5]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return (grad_input, grad_weight_r, grad_weight_i, grad_weight_j,
            grad_weight_k, grad_bias)


class QuaternionLinear(Module):
    """Applies a quaternion linear transformation to the incoming data.
    """

    def __init__(self, in_features, out_features, bias=True, init_criterion
        ='glorot', weight_init='quaternion', seed=None):
        super(QuaternionLinear, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.
            out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.
            out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.
            out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.
            out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features * 4))
        else:
            self.register_parameter('bias', None)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init, 'unitary': unitary_init}[self
            .weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.
            k_weight, winit, self.rng, self.init_criterion)

    def forward(self, input):
        if input.dim() == 3:
            T, N, C = input.size()
            input = input.contiguous().view(T * N, C)
            output = QuaternionLinearFunction.apply(input, self.r_weight,
                self.i_weight, self.j_weight, self.k_weight, self.bias)
            output = output.view(T, N, output.size(1))
        elif input.dim() == 2:
            output = QuaternionLinearFunction.apply(input, self.r_weight,
                self.i_weight, self.j_weight, self.k_weight, self.bias)
        else:
            raise NotImplementedError
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.
            in_features) + ', out_features=' + str(self.out_features
            ) + ', bias=' + str(self.bias is not None
            ) + ', init_criterion=' + str(self.init_criterion
            ) + ', weight_init=' + str(self.weight_init) + ', seed=' + str(self
            .seed) + ')'


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
