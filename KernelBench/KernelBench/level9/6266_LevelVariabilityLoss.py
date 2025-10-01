import torch
import torch.nn as nn


class LevelVariabilityLoss(nn.Module):
    """Computes the variability penalty for the level.
  levels: levels obtained from exponential smoothing component of ESRNN.
          tensor with shape (batch, n_time).
  level_variability_penalty: float.
  return: level_var_loss
  """

    def __init__(self, level_variability_penalty):
        super(LevelVariabilityLoss, self).__init__()
        self.level_variability_penalty = level_variability_penalty

    def forward(self, levels):
        assert levels.shape[1] > 2
        level_prev = torch.log(levels[:, :-1])
        level_next = torch.log(levels[:, 1:])
        log_diff_of_levels = torch.sub(level_prev, level_next)
        log_diff_prev = log_diff_of_levels[:, :-1]
        log_diff_next = log_diff_of_levels[:, 1:]
        diff = torch.sub(log_diff_prev, log_diff_next)
        level_var_loss = diff ** 2
        level_var_loss = level_var_loss.mean() * self.level_variability_penalty
        return level_var_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'level_variability_penalty': 4}]
