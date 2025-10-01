import torch
import torch.nn as nn
import torch.cuda
import torch.distributed


class SimpleFusionGenerator(nn.Module):

    def __init__(self, decoder_input_size, lm_input_size, output_size):
        super(SimpleFusionGenerator, self).__init__()
        self.decoder_linear = nn.Linear(decoder_input_size, output_size)
        self.lm_linear = nn.Linear(lm_input_size, output_size, bias=False)
        self.gen_func = nn.LogSoftmax(dim=-1)

    def forward(self, decoder_hidden, lm_hidden):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.

        Args:
           decoder_hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
           lm_hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
        """
        decoder_logits = self.decoder_linear(decoder_hidden)
        lm_logits = self.lm_linear(lm_hidden)
        logits = (decoder_logits + lm_logits).float()
        log_probs = self.gen_func(logits)
        return log_probs


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'decoder_input_size': 4, 'lm_input_size': 4, 'output_size': 4}
        ]
