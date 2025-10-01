import torch
import torch.nn as nn
import torch.utils.data


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """ Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.

    :param torch.Tensor x:
    :return torch.Tensor:
    """
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int',
        padding_idx: 'int'):
        assert padding_idx is not None
        num_embeddings += padding_idx + 1
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx
            )

    def forward(self, input, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        if use_cache:
            pos = int(self.padding_idx + input.size(1))
            positions = input.data.new(1, 1).fill_(pos)
        else:
            positions = create_position_ids_from_input_ids(input, self.
                padding_idx)
        return super().forward(positions), positions


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_embeddings': 4, 'embedding_dim': 4, 'padding_idx': 4}]
