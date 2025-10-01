import torch
import torch.nn as nn
import torch.utils.checkpoint


class DistanceNetwork(nn.Module):

    def __init__(self, n_feat, p_drop=0.1):
        super(DistanceNetwork, self).__init__()
        self.proj_symm = nn.Linear(n_feat, 37 * 2)
        self.proj_asymm = nn.Linear(n_feat, 37 + 19)
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.zeros_(self.proj_symm.weight)
        nn.init.zeros_(self.proj_asymm.weight)
        nn.init.zeros_(self.proj_symm.bias)
        nn.init.zeros_(self.proj_asymm.bias)

    def forward(self, x):
        logits_asymm = self.proj_asymm(x)
        logits_theta = logits_asymm[:, :, :, :37].permute(0, 3, 1, 2)
        logits_phi = logits_asymm[:, :, :, 37:].permute(0, 3, 1, 2)
        logits_symm = self.proj_symm(x)
        logits_symm = logits_symm + logits_symm.permute(0, 2, 1, 3)
        logits_dist = logits_symm[:, :, :, :37].permute(0, 3, 1, 2)
        logits_omega = logits_symm[:, :, :, 37:].permute(0, 3, 1, 2)
        return logits_dist, logits_omega, logits_theta, logits_phi


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_feat': 4}]
