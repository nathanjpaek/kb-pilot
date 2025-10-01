import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.

    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.

    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e+30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)
    return probs


class OutputLayer(nn.Module):
    """Output Layer which outputs the probability distribution for the answer span in the context span.
    Takes inputs from 2 Model Encoder Layers.
    """

    def __init__(self, drop_prob, word_embed):
        """
        @param drop_prob (float): Probability of zero-ing out activations.
        @param word_embed (int): Word vector size. (128)
        """
        super(OutputLayer, self).__init__()
        self.ff = nn.Linear(2 * word_embed, 1)

    def forward(self, input_1, input_2, mask):
        """Encodes the word embeddings.
        @param input_1 (torch.Tensor): Word vectors from first Model Encoder Layer. (batch_size, hidden_size, sent_len)
        @param input_2 (torch.Tensor): Word vectors from second Model Encoder Layer. (batch_size, hidden_size, sent_len)
        @returns p (torch.Tensor): Probability distribution for start/end token. (batch_size, sent_len)
        """
        x = torch.cat((input_1, input_2), dim=1)
        x = self.ff(x.permute(0, 2, 1)).permute(0, 2, 1)
        logits = x.squeeze()
        log_p = masked_softmax(logits, mask, log_softmax=True)
        return log_p


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'drop_prob': 4, 'word_embed': 4}]
