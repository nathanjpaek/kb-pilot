from _paritybench_helpers import _mock_config
import torch
import typing
from torch import nn


def create_sinusoidal_embeddings(n_pos, dim, out):
    out.requires_grad = False
    positions = torch.arange(0, n_pos)[:, None]
    dimensions = torch.arange(0, dim)
    position_enc = positions / torch.pow(10000, 2 * (dimensions // 2) / dim)
    out[:, 0::2] = torch.sin(position_enc[:, 0::2])
    out[:, 1::2] = torch.cos(position_enc[:, 1::2])


class GeCEmbeddings(nn.Module):
    """Construct the embeddings from gene, (strand and spacing embeddings).
    """

    def __init__(self, config: 'BioConfig', position_embeddings: 'bool'=True):
        super().__init__()
        self.generep_embeddings = nn.Linear(config.input_rep_size, config.
            hidden_size)
        if position_embeddings:
            self.position_embeddings: 'nn.Embedding' = nn.Embedding(config.
                max_position_embeddings, config.hidden_size)
            if config.sinusoidal_pos_embds:
                create_sinusoidal_embeddings(n_pos=config.
                    max_position_embeddings, dim=config.hidden_size, out=
                    self.position_embeddings.weight)
        self.direction_embeddings: 'nn.Embedding' = nn.Embedding(3, config.
            hidden_size)
        self.length_embeddings: 'nn.Embedding' = nn.Embedding(config.
            gene_max_length // config.gene_length_bin_size + 1, config.
            hidden_size)
        self.gene_length_bin_size = config.gene_length_bin_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, gene_reps: 'torch.Tensor', strands:
        'typing.Optional[torch.Tensor]'=None, lengths:
        'typing.Optional[torch.Tensor]'=None, **kwargs) ->torch.Tensor:
        if strands is None:
            strands = torch.zeros_like(gene_reps[:, :, 0], dtype=torch.long)
        else:
            strands = strands.long()
        if lengths is None:
            lengths = torch.ones_like(gene_reps[:, :, 0], dtype=torch.long)
        else:
            lengths = strands.long()
        generep_embeddings = self.generep_embeddings(gene_reps)
        direction_embeddings = self.direction_embeddings(strands + 1)
        length_embeddings = self.length_embeddings(torch.clamp(lengths, 1,
            self.length_embeddings.num_embeddings) // self.gene_length_bin_size
            )
        embeddings = (generep_embeddings + direction_embeddings +
            length_embeddings)
        if hasattr(self, 'position_embeddings'):
            position_ids = torch.arange(gene_reps.size()[1], dtype=torch.
                long, device=gene_reps.device)
            position_ids = position_ids.unsqueeze(0).expand(gene_reps.shape
                [:-1])
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(input_rep_size=4, hidden_size=4,
        max_position_embeddings=4, sinusoidal_pos_embds=4, gene_max_length=
        4, gene_length_bin_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}]
