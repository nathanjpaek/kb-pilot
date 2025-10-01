from _paritybench_helpers import _mock_config
import torch
import torch.utils.data
import torch.utils
import torch.utils.checkpoint


class Joiner(torch.nn.Module):

    def __init__(self, config):
        super(Joiner, self).__init__()
        self.tanh = torch.nn.Tanh()
        self.num_outputs = config.num_tokens + 1
        self.blank_index = 0
        self.linear = torch.nn.Linear(config.num_joiner_hidden, self.
            num_outputs)

    def forward(self, encoder_out, decoder_out):
        combined = encoder_out.unsqueeze(2) + decoder_out.unsqueeze(1)
        out = self.tanh(combined)
        out = self.linear(out).log_softmax(3)
        return out

    def forward_one_step(self, encoder_out, decoder_out):
        combined = encoder_out + decoder_out
        out = self.tanh(combined)
        out = self.linear(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(num_tokens=4, num_joiner_hidden=4)}]
