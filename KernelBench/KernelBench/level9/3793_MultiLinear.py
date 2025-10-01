import torch
import numpy as np
import torch.nn as nn


def tensor(x, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.type(dtype)
    x = torch.tensor(x, device=Config.DEVICE, dtype=dtype)
    return x


def batch_linear(input, weight, bias=None):
    """ input: (N, D), weight: (N, D, H), bias: (N, H) """
    if bias is not None:
        return torch.bmm(input.unsqueeze(1), weight).squeeze(1) + bias
    else:
        return torch.bmm(input.unsqueeze(1), weight).squeeze(1)


def weight_init(weight, w_scale=1.0):
    init_f = nn.init.orthogonal_
    init_f(weight.data)
    weight.data.mul_(w_scale)
    return weight


class BaseNormalizer:

    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return


class RescaleNormalizer(BaseNormalizer):

    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)
        self.coef = coef

    def __call__(self, x):
        x = np.asarray(x)
        return self.coef * x


class Config:
    DEVICE = torch.device('cpu')

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.task_fn = None
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.network_fn = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.replay_fn = None
        self.random_process_fn = None
        self.discount = None
        self.target_network_update_freq = None
        self.exploration_steps = None
        self.logger = None
        self.history_length = None
        self.double_q = False
        self.tag = 'vanilla'
        self.num_workers = 1
        self.gradient_clip = None
        self.entropy_weight = 0
        self.use_gae = False
        self.gae_tau = 1.0
        self.target_network_mix = 0.001
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
        self.min_memory_size = None
        self.max_steps = 0
        self.rollout_length = None
        self.value_loss_weight = 1.0
        self.iteration_log_interval = 30
        self.categorical_v_min = None
        self.categorical_v_max = None
        self.categorical_n_atoms = 51
        self.num_quantiles = None
        self.optimization_epochs = 4
        self.mini_batch_size = 64
        self.termination_regularizer = 0
        self.sgd_update_frequency = None
        self.random_action_prob = None
        self.__eval_env = None
        self.log_interval = int(1000.0)
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 10
        self.async_actor = True
        self.abs_dim = 512

    @property
    def eval_env(self):
        return self.__eval_env

    @eval_env.setter
    def eval_env(self, env):
        self.__eval_env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.task_name = env.name

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])


class MultiLinear(nn.Module):

    def __init__(self, input_dim, output_dim, n_heads, key, w_scale=1.0):
        super().__init__()
        self.weights = nn.Parameter(weight_init(torch.randn(n_heads,
            input_dim, output_dim), w_scale=w_scale))
        self.biases = nn.Parameter(torch.zeros(n_heads, output_dim))
        self.key = key

    def forward(self, inputs, info):
        weights = self.weights[tensor(info[self.key], torch.int64), :, :]
        biases = self.biases[tensor(info[self.key], torch.int64), :]
        return batch_linear(inputs, weight=weights, bias=biases)

    def get_weight(self, info):
        return self.weights[tensor(info[self.key], torch.int64), :, :]

    def load_weight(self, weight_dict):
        for i, weight in weight_dict.items():
            self.weights.data[i] = weight


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([5, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'n_heads': 4, 'key': 4}]
