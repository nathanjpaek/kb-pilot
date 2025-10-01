import torch
import torch.nn as nn
import torch.utils.data


class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, input_hidden_traces, target_hidden_traces):
        Attn = torch.bmm(target_hidden_traces, input_hidden_traces.
            transpose(1, 2))
        Attn_size = Attn.size()
        Attn = Attn - Attn.max(2)[0].unsqueeze(2).expand(Attn_size)
        exp_Attn = torch.exp(Attn)
        Attn = exp_Attn / exp_Attn.sum(2).unsqueeze(2).expand(Attn_size)
        return Attn


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
