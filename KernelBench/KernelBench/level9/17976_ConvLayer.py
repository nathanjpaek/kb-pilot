from _paritybench_helpers import _mock_config
import torch
import torch.nn.functional as F
import torch.nn as nn


class ConvLayer(nn.Module):
    """Conv layer for qa output"""

    def __init__(self, config):
        """
        Args:
            config (ModelArguments): ModelArguments
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=config.hidden_size, out_channels
            =config.qa_conv_out_channel, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=config.hidden_size, out_channels
            =config.qa_conv_out_channel, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=config.hidden_size, out_channels
            =config.qa_conv_out_channel, kernel_size=5, padding=2)
        self.drop_out = nn.Dropout(0.3)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): Layer input

        Returns:
            torch.Tensor: output of conv layer (batch_size * qa_conv_out_channel x 3 * max_seq_legth)
        """
        conv_input = x.transpose(1, 2)
        conv_output1 = F.relu(self.conv1(conv_input))
        conv_output3 = F.relu(self.conv3(conv_input))
        conv_output5 = F.relu(self.conv5(conv_input))
        concat_output = torch.cat((conv_output1, conv_output3, conv_output5
            ), dim=1)
        concat_output = concat_output.transpose(1, 2)
        concat_output = self.drop_out(concat_output)
        return concat_output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, qa_conv_out_channel=4)}]
