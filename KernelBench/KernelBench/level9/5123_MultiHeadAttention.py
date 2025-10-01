import math
import torch
import torch.nn as nn


def dot_scaled_attention(query: 'torch.Tensor', key: 'torch.Tensor', value:
    'torch.Tensor'):
    """ Dot scaled attention
	Implement dot-product scaled attention which takes query, key, value and gives attention scores.

	Arguments:
	query -- Query tensor
				in shape (sequence_length, batch_size, d_k)
	key -- Key tensor
				in shape (sequence_length, batch_size, d_k)
	value -- Value tensor
				in shape (sequence_length, batch_size, d_k)
	padding_mask -- Padding mask tensor in torch.bool type
				in shape (sequence_length, batch_size)
				True for <PAD>, False for non-<PAD>

	Returns:
	attention -- Attention result tensor
				in shape (sequence_length, batch_size, d_k)
	"""
    assert query.shape == key.shape == value.shape
    query_shape = query.shape
    _seq_len, _, d_k = query_shape
    QK_t_scaled = torch.bmm(key.permute(1, 0, 2), query.permute(1, 2, 0)
        ) / math.sqrt(d_k)
    distribution = nn.functional.softmax(QK_t_scaled, dim=1)
    attention = torch.bmm(value.permute(1, 2, 0), distribution).permute(2, 0, 1
        )
    assert attention.shape == query_shape
    return attention, distribution


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_dim: 'int', n_head: 'int'=4):
        """ Multi-head attention initializer
		Use below attributes to implement the forward function

		Attributes:
		n_head -- the number of heads
		d_k -- Hidden dimension of the dot scaled attention
		V_linear -- Linear function to project hidden_dim of value to d_k
		K_linear -- Linear function to project hidden_dim of key to d_k
		Q_linear -- Linear function to project hidden_dim of query to d_k
		O_linear -- Linear function to project collections of d_k to hidden_dim
		"""
        super().__init__()
        assert hidden_dim % n_head == 0
        self.n_head = n_head
        self.d_k = hidden_dim // n_head
        self.V_linear = nn.Linear(hidden_dim, self.n_head * self.d_k, bias=
            False)
        self.K_linear = nn.Linear(hidden_dim, self.n_head * self.d_k, bias=
            False)
        self.Q_linear = nn.Linear(hidden_dim, self.n_head * self.d_k, bias=
            False)
        self.O_linear = nn.Linear(self.n_head * self.d_k, hidden_dim, bias=
            False)

    def forward(self, value: 'torch.Tensor', key: 'torch.Tensor', query:
        'torch.Tensor'):
        """ Multi-head attention forward function
		Implement multi-head attention which takes value, key, query, and gives attention score.
		Use dot-scaled attention you have implemented above.

		Note: If you adjust the dimension of batch_size dynamically,
			  you can implement this function without any iteration.

		Parameters:
		value -- Value tensor
					in shape (sequence_length, batch_size, hidden_dim)
		key -- Key tensor
					in shape (sequence_length, batch_size, hidden_dim)
		query -- Query tensor
					in shape (sequence_length, batch_size, hidden_dim)

		Returns:
		attention -- Attention result tensor
					in shape (sequence_length, batch_size, hidden_dim)
		"""
        assert value.shape == key.shape == query.shape
        input_shape = value.shape
        _seq_length, batch_size, _hidden_dim = input_shape
        Q_embed_concat = torch.cat(self.Q_linear(query.permute(1, 0, 2)).
            split(self.d_k, dim=2), 0).permute(1, 0, 2)
        K_embed_concat = torch.cat(self.K_linear(key.permute(1, 0, 2)).
            split(self.d_k, dim=2), 0).permute(1, 0, 2)
        V_embed_concat = torch.cat(self.V_linear(value.permute(1, 0, 2)).
            split(self.d_k, dim=2), 0).permute(1, 0, 2)
        attention_stacked, distribution = dot_scaled_attention(query=
            Q_embed_concat, key=K_embed_concat, value=V_embed_concat)
        attention = self.O_linear(torch.cat(attention_stacked.permute(1, 0,
            2).split(batch_size, dim=0), 2)).permute(1, 0, 2)
        assert attention.shape == input_shape
        return attention, distribution


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'hidden_dim': 4}]
