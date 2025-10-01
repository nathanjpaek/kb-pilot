import torch
import torch.nn as nn


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, query_dim, context_dim, attention_type='general'):
        super(Attention, self).__init__()
        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')
        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(query_dim, query_dim, bias=False)
        if query_dim != context_dim:
            self.linear_proj = nn.Linear(query_dim, context_dim, bias=False)
        self.linear_out = nn.Linear(context_dim * 2, context_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, query_dim = query.size()
        batch_size, query_len, context_dim = context.size()
        if self.attention_type == 'general':
            query = query.reshape(batch_size * output_len, query_dim)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, query_dim)
        if query_dim != context_dim:
            query = self.linear_proj(query)
        attention_scores = torch.bmm(query, context.transpose(1, 2).
            contiguous())
        attention_scores = attention_scores.view(batch_size * output_len,
            query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len,
            query_len)
        mix = torch.bmm(attention_weights, context)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * context_dim)
        output = self.linear_out(combined).view(batch_size, output_len,
            context_dim)
        output = self.tanh(output)
        attention_weights = attention_weights.mean(dim=1)
        return output, attention_weights


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'query_dim': 4, 'context_dim': 4}]
