import math
import torch
import numpy as np
import torch.nn.functional as F


def truncated_standardized_normal(shape, a=-2.0, b=2.0):
    a = torch.Tensor([a])
    b = torch.Tensor([b])
    U = torch.distributions.uniform.Uniform(0, 1)
    u = U.sample(shape)
    Fa = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
    Fb = 0.5 * (1 + torch.erf(b / math.sqrt(2)))
    return math.sqrt(2) * torch.erfinv(2 * ((Fb - Fa) * u + Fa) - 1)


def get_affine_params(ensemble_size, in_features, out_features):
    w = truncated_standardized_normal(shape=(ensemble_size, in_features,
        out_features)) / (2.0 * math.sqrt(in_features))
    w = torch.nn.Parameter(w)
    b = torch.nn.Parameter(torch.zeros(ensemble_size, 1, out_features,
        dtype=torch.float32))
    return w, b


class EnsembleModel(torch.nn.Module):

    def __init__(self, ensemble_size, input_num, output_num, hidden_num=200):
        super().__init__()
        self.num_nets = ensemble_size
        self.input_num = input_num
        self.output_num = output_num
        self.lin0_w, self.lin0_b = get_affine_params(ensemble_size,
            input_num, hidden_num)
        self.lin1_w, self.lin1_b = get_affine_params(ensemble_size,
            hidden_num, hidden_num)
        self.lin2_w, self.lin2_b = get_affine_params(ensemble_size,
            hidden_num, hidden_num)
        self.lin3_w, self.lin3_b = get_affine_params(ensemble_size,
            hidden_num, hidden_num)
        self.lin4_w, self.lin4_b = get_affine_params(ensemble_size,
            hidden_num, hidden_num)
        self.lin5_w, self.lin5_b = get_affine_params(ensemble_size,
            hidden_num, hidden_num)
        self.lin6_w, self.lin6_b = get_affine_params(ensemble_size,
            hidden_num, 2 * output_num)
        self.inputs_mu = torch.nn.Parameter(torch.zeros(input_num),
            requires_grad=False)
        self.inputs_sigma = torch.nn.Parameter(torch.zeros(input_num),
            requires_grad=False)
        self.max_logvar = torch.nn.Parameter(torch.ones(1, output_num,
            dtype=torch.float32) / 2.0)
        self.min_logvar = torch.nn.Parameter(-torch.ones(1, output_num,
            dtype=torch.float32) * 10.0)

    def compute_decays(self):
        loss = 0.0
        loss += 1.0 * (self.lin0_w ** 2).sum()
        loss += 1.0 * (self.lin1_w ** 2).sum()
        loss += 1.0 * (self.lin2_w ** 2).sum()
        loss += 1.0 * (self.lin3_w ** 2).sum()
        loss += 1.0 * (self.lin4_w ** 2).sum()
        loss += 1.0 * (self.lin5_w ** 2).sum()
        loss += 1.0 * (self.lin6_w ** 2).sum()
        return 1e-05 * loss / 2.0

    def fit_input_stats(self, data):
        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0
        self.inputs_mu.data = torch.from_numpy(mu).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).float()

    def forward(self, inputs):
        inputs = (inputs - self.inputs_mu) / self.inputs_sigma
        inputs = inputs.matmul(self.lin0_w) + self.lin0_b
        inputs = F.silu(inputs)
        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = F.silu(inputs)
        inputs = inputs.matmul(self.lin2_w) + self.lin2_b
        inputs = F.silu(inputs)
        inputs = inputs.matmul(self.lin3_w) + self.lin3_b
        inputs = F.silu(inputs)
        inputs = inputs.matmul(self.lin4_w) + self.lin4_b
        inputs = F.silu(inputs)
        inputs = inputs.matmul(self.lin5_w) + self.lin5_b
        inputs = F.silu(inputs)
        inputs = inputs.matmul(self.lin6_w) + self.lin6_b
        mean = inputs[:, :, :self.output_num]
        logvar = inputs[:, :, self.output_num:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'ensemble_size': 4, 'input_num': 4, 'output_num': 4}]
