import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeStrech(nn.Module):

    def __init__(self, scale):
        super(TimeStrech, self).__init__()
        self.scale = scale

    def forward(self, x):
        mel_size = x.size(-1)
        x = F.interpolate(x, scale_factor=(1, self.scale), align_corners=
            False, recompute_scale_factor=True, mode='bilinear').squeeze()
        if x.size(-1) < mel_size:
            noise_length = mel_size - x.size(-1)
            random_pos = random.randint(0, x.size(-1)) - noise_length
            if random_pos < 0:
                random_pos = 0
            noise = x[..., random_pos:random_pos + noise_length]
            x = torch.cat([x, noise], dim=-1)
        else:
            x = x[..., :mel_size]
        return x.unsqueeze(1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale': 1.0}]
