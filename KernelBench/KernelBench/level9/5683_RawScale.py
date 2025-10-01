import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class RawScale(torch.nn.Module):
    """
    Scale raw data to [-1, 1] in a symmetric manner, which meets bipolar/unipolar bitstream requirements.
    The remaining data count for 'quantile' quantile of the total data.
    The input quantile needs to be within (0, 1].
    """

    def __init__(self, hwcfg={'quantile': 1}):
        super(RawScale, self).__init__()
        self.hwcfg = {}
        self.hwcfg['quantile'] = hwcfg['quantile']
        assert hwcfg['quantile'] > 0 and hwcfg['quantile'
            ] <= 1, "Error: the hw config 'quantile' of " + str(self
            ) + ' class needs to be within (0, 1].'
        self.quantile = hwcfg['quantile']
        self.quantile_lower = 0.5 - self.quantile / 2
        self.quantile_upper = 0.5 + self.quantile / 2

    def forward(self, raw):
        lower_bound = torch.quantile(raw, self.quantile_lower)
        upper_bound = torch.quantile(raw, self.quantile_upper)
        scale = torch.max(lower_bound.abs(), upper_bound.abs())
        output = raw.clamp(lower_bound, upper_bound).div(scale)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
