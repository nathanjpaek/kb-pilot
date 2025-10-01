from _paritybench_helpers import _mock_config
import torch
from torch import nn


class CrossEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(CrossEmbeddings, self).__init__()
        self.position_embeddings = nn.Embedding(config.
            max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, concat_embeddings, concat_type=None):
        _batch_size, seq_length = concat_embeddings.size(0
            ), concat_embeddings.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=
            concat_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(concat_embeddings.
            size(0), -1)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = concat_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(max_position_embeddings=4,
        hidden_size=4, hidden_dropout_prob=0.5)}]
