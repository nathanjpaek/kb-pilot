from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class ProteinBertPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.trainable_encoder = config.trainable_encoder
        if self.trainable_encoder:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        if self.trainable_encoder:
            first_token_tensor = hidden_states[:, 0]
            pooled_output = self.dense(first_token_tensor)
        else:
            mean_token_tensor = hidden_states.mean(dim=1)
            pooled_output = self.dense(mean_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(trainable_encoder=False,
        hidden_size=4)}]
