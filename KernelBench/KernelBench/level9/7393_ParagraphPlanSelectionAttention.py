import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import torch.distributed


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments
        ), 'Not all arguments have the same value: ' + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths
        ).repeat(batch_size, 1).lt(lengths.unsqueeze(1))


class ParagraphPlanSelectionAttention(nn.Module):

    def __init__(self, dim):
        super(ParagraphPlanSelectionAttention, self).__init__()
        self.dim = dim

    def score(self, h_t, h_s):
        src_batch, _src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(src_dim, self.dim)
        h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
        h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
        h_s_ = h_s.transpose(1, 2)
        return torch.bmm(h_t, h_s_)

    def forward(self, source, memory_bank, memory_lengths=None):
        """
        Args
        :param source (FloatTensor): query vectors ``(batch, tgt_len, dim)``
        :param memory_bank (FloatTensor): source vectors ``(batch, src_len, dim)``
        :param memory_lengths (LongTensor): the source context lengths ``(batch,)``
        :return:
        """
        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        align = self.score(source, memory_bank)
        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)
            align.masked_fill_(~mask, -float('inf'))
        align_vectors = F.log_softmax(align.view(batch * target_l, source_l
            ), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)
        return align_vectors


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
