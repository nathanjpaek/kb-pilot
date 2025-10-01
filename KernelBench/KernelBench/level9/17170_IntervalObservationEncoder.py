import torch
from torch import nn


class IntervalObservationEncoder(nn.Module):

    def __init__(self, num_input_channel: 'int', num_output_channel: 'int',
        kernel_size: 'int', initial_output_weight_value: 'float'):
        super().__init__()
        assert initial_output_weight_value <= 1
        self.conv_1d = nn.Conv1d(in_channels=num_input_channel,
            out_channels=num_output_channel, kernel_size=kernel_size, stride=1)
        self.weight = nn.Parameter(torch.tensor(initial_output_weight_value
            ).type(torch.FloatTensor), requires_grad=True)

    def forward(self, observation: 'torch.Tensor') ->torch.Tensor:
        batch_size, _window_size, channel_size, _ = observation.shape
        interval_feature = torch.transpose(observation, 1, 2)
        interval_feature = interval_feature.reshape(batch_size,
            channel_size, -1)
        interval_feature = self.conv_1d(interval_feature)
        interval_feature = torch.squeeze(interval_feature, dim=-1)
        interval_feature = interval_feature * self.weight
        return interval_feature


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_input_channel': 4, 'num_output_channel': 4,
        'kernel_size': 4, 'initial_output_weight_value': 1}]
