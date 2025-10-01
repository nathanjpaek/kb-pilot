import torch
import torch.utils.data
import torch.nn as nn


class AdditiveAttention(nn.Module):

    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.attention_w1 = nn.Linear(enc_hidden_dim, enc_hidden_dim)
        self.attention_w2 = nn.Linear(dec_hidden_dim, enc_hidden_dim)
        self.attention_v = nn.Linear(enc_hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.enc_hidden_dim = enc_hidden_dim
        self.should_print = False
        self.att_mat = []

    def forward(self, encoder_outputs, decoder_output):
        w2_decoder_output = self.attention_w2(decoder_output)
        w1_transformed_encoder_outputs = self.attention_w1(encoder_outputs)
        w1_w2_sum = w1_transformed_encoder_outputs + w2_decoder_output
        w1_w2_sum_tanh = w1_w2_sum.tanh()
        attention_weights = self.attention_v(w1_w2_sum_tanh)
        softmax_attention_weights = self.softmax(attention_weights.squeeze(2))
        if self.should_print:
            to_cpu = softmax_attention_weights.cpu()
            row = to_cpu[0].data.numpy()
            self.att_mat.append(row)
        weighted_encoder_outputs = (encoder_outputs *
            softmax_attention_weights.unsqueeze(2).expand(-1, -1, self.
            enc_hidden_dim))
        context = weighted_encoder_outputs.sum(dim=1)
        return context


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'enc_hidden_dim': 4, 'dec_hidden_dim': 4}]
