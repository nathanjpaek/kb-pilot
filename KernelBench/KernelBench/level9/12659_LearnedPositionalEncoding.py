import torch
import torch.nn as nn
import torch.cuda
import torch.distributed


class LearnedPositionalEncoding(nn.Module):

    def __init__(self, context_size, embedding_dim, dropout=0):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(context_size, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, emb, step=None, offset=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """
        if step is None:
            position_ids = torch.arange(0, emb.shape[0], dtype=torch.long,
                device=emb.device)
        else:
            position_ids = torch.arange(step, step + 1, dtype=torch.long,
                device=emb.device)
        position_ids = position_ids.unsqueeze(1).repeat(1, emb.shape[1])
        if offset is not None:
            offset = offset.unsqueeze(0)
            position_ids += offset
        pe_vals = self.pe(position_ids)
        emb = emb + pe_vals
        emb = self.dropout(emb)
        return emb


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'context_size': 4, 'embedding_dim': 4}]
