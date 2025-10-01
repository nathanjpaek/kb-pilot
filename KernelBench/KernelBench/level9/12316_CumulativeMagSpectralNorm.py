import torch
import torch.nn as nn


class CumulativeMagSpectralNorm(nn.Module):

    def __init__(self, cumulative=False, use_mid_freq_mu=False):
        """

        Args:
            cumulative: 是否采用累积的方式计算 mu
            use_mid_freq_mu: 仅采用中心频率的 mu 来代替全局 mu

        Notes:
            先算均值再累加 等同于 先累加再算均值

        """
        super().__init__()
        self.eps = 1e-06
        self.cumulative = cumulative
        self.use_mid_freq_mu = use_mid_freq_mu

    def forward(self, input):
        assert input.ndim == 4, f'{self.__name__} only support 4D input.'
        batch_size, n_channels, n_freqs, n_frames = input.size()
        device = input.device
        data_type = input.dtype
        input = input.reshape(batch_size * n_channels, n_freqs, n_frames)
        if self.use_mid_freq_mu:
            step_sum = input[:, int(n_freqs // 2 - 1), :]
        else:
            step_sum = torch.mean(input, dim=1)
        if self.cumulative:
            cumulative_sum = torch.cumsum(step_sum, dim=-1)
            entry_count = torch.arange(1, n_frames + 1, dtype=data_type,
                device=device)
            entry_count = entry_count.reshape(1, n_frames)
            entry_count = entry_count.expand_as(cumulative_sum)
            mu = cumulative_sum / entry_count
            mu = mu.reshape(batch_size * n_channels, 1, n_frames)
        else:
            mu = torch.mean(step_sum, dim=-1)
            mu = mu.reshape(batch_size * n_channels, 1, 1)
        input_normed = input / (mu + self.eps)
        input_normed = input_normed.reshape(batch_size, n_channels, n_freqs,
            n_frames)
        return input_normed


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
