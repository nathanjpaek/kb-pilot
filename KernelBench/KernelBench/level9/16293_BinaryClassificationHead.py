from _paritybench_helpers import _mock_config
import torch


class BinaryClassificationHead(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(config.hidden_size, 1)

    def init_weights(self):
        self.dense.weight.data.normal_(mean=0.0, std=self.config.
            initializer_range)
        if self.dense.bias is not None:
            self.dense.bias.data.zero_()

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, hidden_dropout_prob=
        0.5)}]
