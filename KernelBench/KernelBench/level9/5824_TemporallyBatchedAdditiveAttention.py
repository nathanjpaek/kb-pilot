import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):

    def __init__(self, encoder_hidden_state_dim, decoder_hidden_state_dim,
        internal_dim=None):
        super(AdditiveAttention, self).__init__()
        if internal_dim is None:
            internal_dim = int((encoder_hidden_state_dim +
                decoder_hidden_state_dim) / 2)
        self.w1 = nn.Linear(encoder_hidden_state_dim, internal_dim, bias=False)
        self.w2 = nn.Linear(decoder_hidden_state_dim, internal_dim, bias=False)
        self.v = nn.Linear(internal_dim, 1, bias=False)

    def score(self, encoder_state, decoder_state):
        return self.v(torch.tanh(self.w1(encoder_state) + self.w2(
            decoder_state)))

    def forward(self, encoder_states, decoder_state):
        score_vec = torch.cat([self.score(encoder_states[:, i],
            decoder_state) for i in range(encoder_states.shape[1])], dim=1)
        attention_probs = torch.unsqueeze(F.softmax(score_vec, dim=1), dim=2)
        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        return final_context_vec, attention_probs


class TemporallyBatchedAdditiveAttention(AdditiveAttention):

    def __init__(self, encoder_hidden_state_dim, decoder_hidden_state_dim,
        internal_dim=None):
        super(TemporallyBatchedAdditiveAttention, self).__init__(
            encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim)

    def score(self, encoder_state, decoder_state):
        return self.v(torch.tanh(self.w1(encoder_state) + torch.unsqueeze(
            self.w2(decoder_state), dim=1)))

    def forward(self, encoder_states, decoder_state):
        score_vec = self.score(encoder_states, decoder_state)
        attention_probs = F.softmax(score_vec, dim=1)
        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        return final_context_vec, torch.squeeze(torch.transpose(
            attention_probs, 1, 2), dim=3)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'encoder_hidden_state_dim': 4, 'decoder_hidden_state_dim': 4}]
