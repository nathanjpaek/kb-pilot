from _paritybench_helpers import _mock_config
import torch
from torch import nn


class PositionEmbedding(nn.Module):
    """
    adpated from transformers package by huggingface.
    """

    def __init__(self, config):
        super(PositionEmbedding, self).__init__()
        self.config = config
        self.pos_embs = nn.Embedding(config['trans_max_pos'], config[
            'trans_hidden'])
        self.LayerNorm = nn.LayerNorm(config['trans_hidden'])
        self.dropout = nn.Dropout(config['trans_drop_prob'])

    def forward(self, input_embs):
        """
        `input_embs` should be shaped as [`numBatch`, `seqLength`, `hiddenSize`]
        """
        seq_length = input_embs.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=
            input_embs.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_embs[:, :, 0])
        position_embeddings = self.pos_embs(position_ids)
        embeddings = input_embs + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(trans_max_pos=4, trans_hidden=4,
        trans_drop_prob=0.5)}]
