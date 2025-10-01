from _paritybench_helpers import _mock_config
import torch
import torch.utils.data


class FBANKNormalizer(torch.nn.Module):

    def __init__(self, config):
        super(FBANKNormalizer, self).__init__()
        self.num_mel_bins = config.num_mel_bins
        self.weight = torch.nn.Parameter(torch.tensor([1 / 10] * self.
            num_mel_bins))
        self.bias = torch.nn.Parameter(torch.tensor([0.0] * self.num_mel_bins))

    def forward(self, fbank):
        out = fbank + self.bias.unsqueeze(0)
        out = out * self.weight.unsqueeze(0)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(num_mel_bins=4)}]
