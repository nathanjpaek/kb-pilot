import torch
import torch.nn as nn


class SequenceSummaryLayer(nn.Module):

    def __init__(self, hidden_size, summary_layers):
        super().__init__()
        self.summary_layers = summary_layers
        self.linear = nn.Linear(hidden_size * summary_layers, hidden_size)
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()

    def forward(self, x):
        stacked_hidden_states = torch.stack(list(x[-self.summary_layers:]),
            dim=-2)
        stacked_hidden_states = stacked_hidden_states[:, 0]
        concat_hidden_states = stacked_hidden_states.view(stacked_hidden_states
            .shape[0], stacked_hidden_states.shape[-2] *
            stacked_hidden_states.shape[-1])
        resized_hidden_states = self.linear(concat_hidden_states)
        pooled_hidden_states = self.pooler_activation(self.pooler(
            resized_hidden_states))
        return pooled_hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'summary_layers': 1}]
