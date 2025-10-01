from _paritybench_helpers import _mock_config
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from numbers import Number
from torch.nn import MSELoss


def isnan(x):
    if isinstance(x, Patches):
        return False
    return torch.isnan(x).any()


class Perturbation:

    def __init__(self):
        pass

    def set_eps(self, eps):
        self.eps = eps

    def concretize(self, x, A, sign=-1, aux=None):
        raise NotImplementedError

    def init(self, x, aux=None, forward=False):
        raise NotImplementedError


class PerturbationL0Norm(Perturbation):

    def __init__(self, eps, x_L=None, x_U=None, ratio=1.0):
        self.eps = eps
        self.x_U = x_U
        self.x_L = x_L
        self.ratio = ratio

    def concretize(self, x, A, sign=-1, aux=None):
        if A is None:
            return None
        eps = math.ceil(self.eps)
        x = x.reshape(x.shape[0], -1, 1)
        center = A.matmul(x)
        x = x.reshape(x.shape[0], 1, -1)
        original = A * x.expand(x.shape[0], A.shape[-2], x.shape[2])
        neg_mask = A < 0
        pos_mask = A >= 0
        if sign == 1:
            A_diff = torch.zeros_like(A)
            A_diff[pos_mask] = A[pos_mask] - original[pos_mask]
            A_diff[neg_mask] = -original[neg_mask]
        else:
            A_diff = torch.zeros_like(A)
            A_diff[pos_mask] = original[pos_mask]
            A_diff[neg_mask] = original[neg_mask] - A[neg_mask]
        A_diff, _ = torch.sort(A_diff, dim=2, descending=True)
        bound = center + sign * A_diff[:, :, :eps].sum(dim=2).unsqueeze(2
            ) * self.ratio
        return bound.squeeze(2)

    def init(self, x, aux=None, forward=False):
        x_L = x
        x_U = x
        if not forward:
            return LinearBound(None, None, None, None, x_L, x_U), x, None
        batch_size = x.shape[0]
        dim = x.reshape(batch_size, -1).shape[-1]
        eye = torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1)
        lw = eye.reshape(batch_size, dim, *x.shape[1:])
        lb = torch.zeros_like(x)
        uw, ub = lw.clone(), lb.clone()
        return LinearBound(lw, lb, uw, ub, x_L, x_U), x, None

    def __repr__(self):
        return 'PerturbationLpNorm(norm=0, eps={})'.format(self.eps)


class PerturbationLpNorm(Perturbation):

    def __init__(self, eps, norm=np.inf, x_L=None, x_U=None):
        if not isinstance(eps, Number):
            if not isinstance(eps, torch.Tensor):
                self.eps = torch.tensor(eps)
            else:
                self.eps = eps
            if len(self.eps.shape) == 1:
                self.eps = torch.diag(self.eps)
            assert self.eps.shape[0] == self.eps.shape[1
                ], 'Argument [eps] must form a n by n square matrix.'
            self.norm = 2
        else:
            self.eps = eps
            self.norm = norm
        self.dual_norm = 1 if norm == np.inf else np.float64(1.0) / (1 - 
            1.0 / self.norm)
        self.x_L = x_L
        self.x_U = x_U
    """Given an variable x and its bound matrix A, compute worst case bound according to Lp norm."""

    def concretize(self, x, A, sign=-1, aux=None):
        if A is None:
            return None

        def concretize_matrix(A):
            nonlocal x
            if not isinstance(A, eyeC):
                A = A.reshape(A.shape[0], A.shape[1], -1)
            if self.norm == np.inf:
                x_L = x - self.eps if self.x_L is None else self.x_L
                x_U = x + self.eps if self.x_U is None else self.x_U
                x_ub = x_U.reshape(x_U.shape[0], -1, 1)
                x_lb = x_L.reshape(x_L.shape[0], -1, 1)
                center = (x_ub + x_lb) / 2.0
                diff = (x_ub - x_lb) / 2.0
                if not isinstance(A, eyeC):
                    bound = A.matmul(center) + sign * A.abs().matmul(diff)
                else:
                    bound = center + sign * diff
            else:
                x = x.reshape(x.shape[0], -1, 1)
                if not isinstance(A, eyeC):
                    if isinstance(self.eps, Number):
                        deviation = A.norm(self.dual_norm, -1) * self.eps
                    else:
                        deviation = A.matmul(self.eps.transpose(0, 1)).norm(
                            self.dual_norm, -1)
                    bound = A.matmul(x) + sign * deviation.unsqueeze(-1)
                elif isinstance(self.eps, Number):
                    bound = x + sign * self.eps
                else:
                    bound = x + sign * self.eps.transpose(0, 1).norm(self.
                        dual_norm, -1)
            bound = bound.squeeze(-1)
            return bound

        def concretize_patches(A):
            nonlocal x
            if self.norm == np.inf:
                x_L = x - self.eps if self.x_L is None else self.x_L
                x_U = x + self.eps if self.x_U is None else self.x_U
                center = (x_U + x_L) / 2.0
                diff = (x_U - x_L) / 2.0
                if not A.identity == 1:
                    unfold_input = F.unfold(center, kernel_size=A.patches.
                        size(-1), padding=A.padding, stride=A.stride
                        ).transpose(-2, -1)
                    unfold_input = unfold_input.view(unfold_input.size(0),
                        unfold_input.size(1), -1, A.patches.size(-3), A.
                        patches.size(-2), A.patches.size(-1))
                    prod = unfold_input * A.patches
                    prod = prod.sum((-1, -2, -3)).transpose(-2, -1)
                    bound = prod.view(prod.size(0), prod.size(1), int(math.
                        sqrt(prod.size(2))), int(math.sqrt(prod.size(2))))
                    unfold_input = F.unfold(diff, kernel_size=A.patches.
                        size(-1), padding=A.padding, stride=A.stride
                        ).transpose(-2, -1)
                    unfold_input = unfold_input.view(unfold_input.size(0),
                        unfold_input.size(1), -1, A.patches.size(-3), A.
                        patches.size(-2), A.patches.size(-1))
                    prod = unfold_input * A.patches.abs()
                    prod = prod.sum((-1, -2, -3)).transpose(-2, -1)
                    bound += sign * prod.view(prod.size(0), prod.size(1),
                        int(math.sqrt(prod.size(2))), int(math.sqrt(prod.
                        size(2))))
                else:
                    bound = center + sign * diff
                return bound
            else:
                x_L = x - self.eps if self.x_L is None else self.x_L
                x_U = x + self.eps if self.x_U is None else self.x_U
                raise NotImplementedError()
        if isinstance(A, eyeC) or isinstance(A, torch.Tensor):
            return concretize_matrix(A)
        elif isinstance(A, Patches):
            return concretize_patches(A)
        elif isinstance(A, BoundList):
            for b in A.bound_list:
                if isinstance(b, eyeC) or isinstance(b, torch.Tensor):
                    pass
        else:
            raise NotImplementedError()

    def init(self, x, aux=None, forward=False):
        if self.norm == np.inf:
            x_L = x - self.eps if self.x_L is None else self.x_L
            x_U = x + self.eps if self.x_U is None else self.x_U
        else:
            x_L = x
            x_U = x
        if not forward:
            return LinearBound(None, None, None, None, x_L, x_U), x, None
        batch_size = x.shape[0]
        dim = x.reshape(batch_size, -1).shape[-1]
        eye = torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1)
        lw = eye.reshape(batch_size, dim, *x.shape[1:])
        lb = torch.zeros_like(x)
        uw, ub = lw.clone(), lb.clone()
        return LinearBound(lw, lb, uw, ub, x_L, x_U), x, None

    def __repr__(self):
        if self.norm == np.inf:
            if self.x_L is None and self.x_U is None:
                return 'PerturbationLpNorm(norm=inf, eps={})'.format(self.eps)
            else:
                return ('PerturbationLpNorm(norm=inf, eps={}, x_L={}, x_U={})'
                    .format(self.eps, self.x_L, self.x_U))
        else:
            return 'PerturbationLpNorm(norm={}, eps={})'.format(self.norm,
                self.eps)


class PerturbationSynonym(Perturbation):

    def __init__(self, budget, eps=1.0, use_simple=False):
        super(PerturbationSynonym, self).__init__()
        self._load_synonyms()
        self.budget = budget
        self.eps = eps
        self.use_simple = use_simple
        self.model = None
        self.train = False

    def __repr__(self):
        return (
            'perturbation(Synonym-based word substitution budget={}, eps={})'
            .format(self.budget, self.eps))

    def _load_synonyms(self, path='data/synonyms.json'):
        with open(path) as file:
            self.synonym = json.loads(file.read())
        logger.info('Synonym list loaded for {} words'.format(len(self.
            synonym)))

    def set_train(self, train):
        self.train = train

    def concretize(self, x, A, sign, aux):
        assert self.model is not None
        x_rep, mask, can_be_replaced = aux
        batch_size, length, dim_word = x.shape[0], x.shape[1], x.shape[2]
        dim_out = A.shape[1]
        max_num_cand = x_rep.shape[2]
        mask_rep = torch.tensor(can_be_replaced, dtype=torch.float32,
            device=A.device)
        num_pos = int(np.max(np.sum(can_be_replaced, axis=-1)))
        update_A = A.shape[-1] > num_pos * dim_word
        if update_A:
            bias = torch.bmm(A, (x * (1 - mask_rep).unsqueeze(-1)).reshape(
                batch_size, -1, 1)).squeeze(-1)
        else:
            bias = 0.0
        A = A.reshape(batch_size, dim_out, -1, dim_word)
        A_new, x_new, x_rep_new, mask_new = [], [], [], []
        zeros_A = torch.zeros(dim_out, dim_word, device=A.device)
        zeros_w = torch.zeros(dim_word, device=A.device)
        zeros_rep = torch.zeros(max_num_cand, dim_word, device=A.device)
        zeros_mask = torch.zeros(max_num_cand, device=A.device)
        for t in range(batch_size):
            cnt = 0
            for i in range(0, length):
                if can_be_replaced[t][i]:
                    if update_A:
                        A_new.append(A[t, :, i, :])
                    x_new.append(x[t][i])
                    x_rep_new.append(x_rep[t][i])
                    mask_new.append(mask[t][i])
                    cnt += 1
            if update_A:
                A_new += [zeros_A] * (num_pos - cnt)
            x_new += [zeros_w] * (num_pos - cnt)
            x_rep_new += [zeros_rep] * (num_pos - cnt)
            mask_new += [zeros_mask] * (num_pos - cnt)
        if update_A:
            A = torch.cat(A_new).reshape(batch_size, num_pos, dim_out, dim_word
                ).transpose(1, 2)
        x = torch.cat(x_new).reshape(batch_size, num_pos, dim_word)
        x_rep = torch.cat(x_rep_new).reshape(batch_size, num_pos,
            max_num_cand, dim_word)
        mask = torch.cat(mask_new).reshape(batch_size, num_pos, max_num_cand)
        length = num_pos
        A = A.reshape(batch_size, A.shape[1], length, -1).transpose(1, 2)
        x = x.reshape(batch_size, length, -1, 1)
        if sign == 1:
            cmp, init = torch.max, -1e+30
        else:
            cmp, init = torch.min, 1e+30
        init_tensor = torch.ones(batch_size, dim_out) * init
        dp = [([init_tensor] * (self.budget + 1)) for i in range(0, length + 1)
            ]
        dp[0][0] = torch.zeros(batch_size, dim_out)
        A = A.reshape(batch_size * length, A.shape[2], A.shape[3])
        Ax = torch.bmm(A, x.reshape(batch_size * length, x.shape[2], x.
            shape[3])).reshape(batch_size, length, A.shape[1])
        Ax_rep = torch.bmm(A, x_rep.reshape(batch_size * length,
            max_num_cand, x.shape[2]).transpose(-1, -2)).reshape(batch_size,
            length, A.shape[1], max_num_cand)
        Ax_rep = Ax_rep * mask.unsqueeze(2) + init * (1 - mask).unsqueeze(2)
        Ax_rep_bound = cmp(Ax_rep, dim=-1).values
        if self.use_simple and self.train:
            return torch.sum(cmp(Ax, Ax_rep_bound), dim=1) + bias
        for i in range(1, length + 1):
            dp[i][0] = dp[i - 1][0] + Ax[:, i - 1]
            for j in range(1, self.budget + 1):
                dp[i][j] = cmp(dp[i - 1][j] + Ax[:, i - 1], dp[i - 1][j - 1
                    ] + Ax_rep_bound[:, i - 1])
        dp = torch.cat(dp[length], dim=0).reshape(self.budget + 1,
            batch_size, dim_out)
        return cmp(dp, dim=0).values + bias

    def init(self, x, aux=None, forward=False):
        tokens, batch = aux
        self.tokens = tokens
        assert len(x.shape) == 3
        batch_size, length, dim_word = x.shape[0], x.shape[1], x.shape[2]
        max_pos = 1
        can_be_replaced = np.zeros((batch_size, length), dtype=np.bool)
        self._build_substitution(batch)
        for t in range(batch_size):
            cnt = 0
            candidates = batch[t]['candidates']
            if tokens[t][0] == '[CLS]':
                candidates = [[]] + candidates + [[]]
            for i in range(len(tokens[t])):
                if tokens[t][i] == '[UNK]' or len(candidates[i]
                    ) == 0 or tokens[t][i] != candidates[i][0]:
                    continue
                for w in candidates[i][1:]:
                    if w in self.model.vocab:
                        can_be_replaced[t][i] = True
                        cnt += 1
                        break
            max_pos = max(max_pos, cnt)
        dim = max_pos * dim_word
        if forward:
            eye = torch.eye(dim_word)
            lw = torch.zeros(batch_size, dim, length, dim_word)
            lb = torch.zeros_like(x)
        word_embeddings = self.model.word_embeddings.weight
        vocab = self.model.vocab
        x_rep = [[[] for i in range(length)] for t in range(batch_size)]
        max_num_cand = 1
        for t in range(batch_size):
            candidates = batch[t]['candidates']
            if tokens[t][0] == '[CLS]':
                candidates = [[]] + candidates + [[]]
            cnt = 0
            for i in range(length):
                if can_be_replaced[t][i]:
                    word_embed = word_embeddings[vocab[tokens[t][i]]]
                    other_embed = x[t, i] - word_embed
                    if forward:
                        lw[t, cnt * dim_word:(cnt + 1) * dim_word, i, :] = eye
                        lb[t, i, :] = torch.zeros_like(word_embed)
                    for w in candidates[i][1:]:
                        if w in self.model.vocab:
                            x_rep[t][i].append(word_embeddings[self.model.
                                vocab[w]] + other_embed)
                    max_num_cand = max(max_num_cand, len(x_rep[t][i]))
                    cnt += 1
                elif forward:
                    lb[t, i, :] = x[t, i, :]
        if forward:
            uw, ub = lw, lb
        else:
            lw = lb = uw = ub = None
        zeros = torch.zeros(dim_word, device=x.device)
        x_rep_, mask = [], []
        for t in range(batch_size):
            for i in range(length):
                x_rep_ += x_rep[t][i] + [zeros] * (max_num_cand - len(x_rep
                    [t][i]))
                mask += [1] * len(x_rep[t][i]) + [0] * (max_num_cand - len(
                    x_rep[t][i]))
        x_rep_ = torch.cat(x_rep_).reshape(batch_size, length, max_num_cand,
            dim_word)
        mask = torch.tensor(mask, dtype=torch.float32, device=x.device
            ).reshape(batch_size, length, max_num_cand)
        x_rep_ = x_rep_ * self.eps + x.unsqueeze(2) * (1 - self.eps)
        inf = 1e+20
        lower = torch.min(mask.unsqueeze(-1) * x_rep_ + (1 - mask).
            unsqueeze(-1) * inf, dim=2).values
        upper = torch.max(mask.unsqueeze(-1) * x_rep_ + (1 - mask).
            unsqueeze(-1) * -inf, dim=2).values
        lower = torch.min(lower, x)
        upper = torch.max(upper, x)
        return LinearBound(lw, lb, uw, ub, lower, upper), x, (x_rep_, mask,
            can_be_replaced)

    def _build_substitution(self, batch):
        for t, example in enumerate(batch):
            if 'candidates' not in example or example['candidates'] is None:
                candidates = []
                tokens = example['sentence'].strip().lower().split(' ')
                for i in range(len(tokens)):
                    _cand = []
                    if tokens[i] in self.synonym:
                        for w in self.synonym[tokens[i]]:
                            if w in self.model.vocab:
                                _cand.append(w)
                    if len(_cand) > 0:
                        _cand = [tokens[i]] + _cand
                    candidates.append(_cand)
                example['candidates'] = candidates


class Interval(tuple):

    def __new__(self, lb=None, ub=None, ptb=None):
        if ub is None:
            assert isinstance(lb, tuple)
            lb, ub = lb
        return tuple.__new__(Interval, (lb, ub))

    def __init__(self, lb, ub, ptb=None):
        if ptb is None:
            self.ptb = None
            assert lb is ub
        elif not isinstance(ptb, Perturbation):
            raise ValueError(
                'ptb must be a Perturbation object or None. Got type {}'.
                format(type(ptb)))
        else:
            self.ptb = ptb

    def __str__(self):
        return '({}, {}) with ptb={}'.format(self[0], self[1], self.ptb)

    def __repr__(self):
        return 'Interval(lb={}, ub={}, ptb={})'.format(self[0], self[1],
            self.ptb)
    """Checking if the other interval is tuple, keep the perturbation."""

    @staticmethod
    def make_interval(lb, ub, other):
        if isinstance(other, Interval):
            return Interval(lb, ub, other.ptb)
        else:
            return lb, ub
    """Given a tuple or Interval object, returns the norm and eps."""

    @staticmethod
    def get_perturbation(interval):
        if isinstance(interval, Interval):
            if isinstance(interval.ptb, PerturbationLpNorm):
                return interval.ptb.norm, interval.ptb.eps
            elif isinstance(interval.ptb, PerturbationSynonym):
                return np.inf, 1.0
            elif isinstance(interval.ptb, PerturbationL0Norm):
                return 0, interval.ptb.eps, interval.ptb.ratio
            elif interval.ptb is None:
                raise RuntimeError(
                    'get_perturbation() encountered an interval that is not perturbed.'
                    )
            else:
                raise RuntimeError(
                    'get_perturbation() does not know how to handle {}'.
                    format(type(interval.ptb)))
        else:
            return np.inf, np.nan
    """Checking if a Interval or tuple object has perturbation enabled."""

    @staticmethod
    def is_perturbed(interval):
        if isinstance(interval, Interval) and interval.ptb is None:
            return False
        else:
            return True


class Bound(nn.Module):

    def __init__(self, input_name, name, ori_name, attr={}, inputs=[],
        output_index=0, options={}, device=None):
        super().__init__()
        self.output_name = []
        (self.input_name, self.name, self.ori_name, self.attr, self.inputs,
            self.output_index, self.options, self.device) = (input_name,
            name, ori_name, attr, inputs, output_index, options, device)
        self.fv = None
        self.from_input = False
        self.bounded = False
        self.IBP_rets = None
        self.perturbed = False
        if options is not None and 'loss_fusion' in options:
            self.loss_fusion = options['loss_fusion']
        else:
            self.loss_fusion = False
    """Check if the i-th input is with perturbation or not."""

    def is_input_perturbed(self, i=0):
        return self.inputs[i].perturbed

    def forward(self, *x):
        raise NotImplementedError

    def interval_propagate(self, *v):
        assert len(v) == 1
        h_L, h_U = v[0]
        return Interval.make_interval(self.forward(h_L), self.forward(h_U),
            v[0])

    def bound_forward(self, dim_in, last):
        raise NotImplementedError

    def bound_backward(self, last_lA, last_uA):
        raise NotImplementedError

    def infer_batch_dim(self, batch_size, *x):
        None
        raise NotImplementedError

    def broadcast_backward(self, A, x):
        shape = x.default_shape
        batch_dim = max(self.batch_dim, 0)
        if isinstance(A, torch.Tensor):
            if x.batch_dim == -1:
                shape = torch.Size([A.shape[batch_dim + 1]] + list(shape))
                dims = []
                cnt_sum = A.ndim - len(shape) - 1
                for i in range(1, A.ndim):
                    if i != self.batch_dim + 1 and cnt_sum > 0:
                        dims.append(i)
                        cnt_sum -= 1
                if dims:
                    A = torch.sum(A, dim=dims)
            else:
                dims = list(range(1, 1 + A.ndim - 1 - len(shape)))
                if dims:
                    A = torch.sum(A, dim=dims)
            dims = []
            for i in range(len(shape)):
                if shape[i] == 1 and A.shape[i + 1] != 1:
                    dims.append(i + 1)
            if dims:
                A = torch.sum(A, dim=dims, keepdim=True)
            assert A.shape[1:] == shape
        elif type(A) == Patches:
            pass
        return A

    @staticmethod
    def broadcast_forward(dim_in, x, shape_res):
        lw, lb, uw, ub = x.lw, x.lb, x.uw, x.ub
        shape_x, shape_res = list(x.lb.shape), list(shape_res)
        if lw is None:
            lw = uw = torch.zeros(dim_in, *shape_x, device=lb.device)
            has_batch_size = False
        else:
            has_batch_size = True
        while len(shape_x) < len(shape_res):
            if not has_batch_size:
                lw, uw = lw.unsqueeze(0), uw.unsqueeze(0)
                lb, ub = lb.unsqueeze(0), ub.unsqueeze(0)
                shape_x = [1] + shape_x
                has_batch_size = True
            else:
                lw, uw = lw.unsqueeze(2), uw.unsqueeze(2)
                lb, ub = lb.unsqueeze(1), ub.unsqueeze(1)
                shape_x = [shape_x[0], 1] + shape_x[1:]
        repeat = [(shape_res[i] // shape_x[i]) for i in range(len(shape_x))]
        lb, ub = lb.repeat(*repeat), ub.repeat(*repeat)
        repeat = repeat[:1] + [1] + repeat[1:]
        lw, uw = lw.repeat(*repeat), uw.repeat(*repeat)
        return lw, lb, uw, ub

    def get_bias(self, A, bias):
        if A is None:
            return 0
        assert not isnan(A)
        assert not isnan(bias)
        if isinstance(A, torch.Tensor):
            if torch.norm(A, p=1) < epsilon:
                return 0
            output_dim = A.shape[0]
            if self.batch_dim != -1:
                batch_size = A.shape[self.batch_dim + 1]
                A_shape = [A.shape[0], np.prod(A.shape[1:self.batch_dim + 1
                    ]).astype(np.int32), batch_size, np.prod(A.shape[self.
                    batch_dim + 2:]).astype(np.int32)]
                A = A.reshape(*A_shape).permute(2, 0, 1, 3).reshape(batch_size,
                    output_dim, -1)
                bias = bias.reshape(*A_shape[1:]).transpose(0, 1).reshape(
                    batch_size, -1, 1)
                bias_new = A.matmul(bias).squeeze(-1).transpose(0, 1)
            else:
                batch_size = A.shape[1]
                A = A.view(output_dim, batch_size, -1)
                bias_new = A.matmul(bias.view(-1))
            if isnan(bias_new):
                return 0
            else:
                return bias_new
        elif type(A) == Patches:
            if torch.norm(A.patches, p=1) < epsilon:
                return 0
            if self.batch_dim != -1:
                batch_size = bias.shape[0]
                bias = F.unfold(bias, kernel_size=A.patches.size(-1),
                    stride=A.stride, padding=A.padding).transpose(-2, -1
                    ).unsqueeze(-2)
                bias.size(1)
                patches = A.patches.view(A.patches.size(0), A.patches.size(
                    1), A.patches.size(-4), A.patches.size(-1) * A.patches.
                    size(-2) * A.patches.size(-3))
                prod = bias * patches
                bias_new = prod.sum(-1).transpose(-2, -1)
                bias_new = bias_new.view(batch_size, bias_new.size(-2), int
                    (math.sqrt(bias_new.size(-1))), int(math.sqrt(bias_new.
                    size(-1))))
            else:
                patches = A.patches
                patches_reshape = torch.sum(patches, dim=(-1, -2, -3)) * bias
                patches_reshape = patches_reshape.transpose(-1, -2)
                return patches_reshape.view(patches_reshape.size(0),
                    patches_reshape.size(1), int(math.sqrt(patches_reshape.
                    size(2))), -1).transpose(0, 1)
            return bias_new
        else:
            return NotImplementedError()


class BoundCos(Bound):

    def __init__(self, input_name, name, ori_name, attr, inputs,
        output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs,
            output_index, options, device)

    def forward(self, x):
        return torch.cos(x)

    def infer_batch_dim(self, batch_size, *x):
        return x[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_name': 4, 'name': 4, 'ori_name': 4, 'attr': 4,
        'inputs': 4, 'output_index': 4, 'options': _mock_config(loss_fusion
        =MSELoss()), 'device': 0}]
