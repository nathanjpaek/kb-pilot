import torch


class SelfAttention(torch.nn.Module):

    def __init__(self, num_heads, model_dim, dropout_keep_prob):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.dropout_keep_prob = dropout_keep_prob
        self.q_layer = torch.nn.Linear(model_dim, model_dim * self.
            num_heads, bias=False)
        self.out_layer = torch.nn.Linear(model_dim * self.num_heads,
            model_dim, bias=False)
        self.out_layer2 = torch.nn.Linear(model_dim * 2, model_dim, bias=False)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(1 - dropout_keep_prob)

    def forward(self, batched_inputs, attn_mask=None):
        q = self._linear_projection(batched_inputs)
        qs = self._split_heads(q)
        tiled_inputs = batched_inputs.unsqueeze(1).repeat(1, self.num_heads,
            1, 1)
        outputs = self._scaled_dot_product(qs, tiled_inputs, tiled_inputs,
            attn_mask)
        outputs = self._concat_heads(outputs)
        if self.num_heads > 1:
            outputs = self.out_layer(outputs)
            outputs = self.relu(outputs)
        outputs = torch.cat([outputs, batched_inputs], dim=-1)
        outputs = self.out_layer2(outputs)
        outputs = self.relu(outputs)
        return outputs

    def _linear_projection(self, batched_inputs):
        q = self.q_layer(batched_inputs)
        return q

    def _split_heads(self, q):

        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            tensor = tensor.view([-1, tensor.size()[1], num_heads, dim])
            return tensor.transpose(1, 2)
        qs = split_last_dimension_then_transpose(q, self.num_heads, self.
            model_dim)
        return qs

    def _scaled_dot_product(self, qs, ks, tiled_inputs, valid_mask):
        queries_dot_keys = torch.matmul(qs, ks.transpose(2, 3))
        scaled_scores = queries_dot_keys
        if valid_mask is not None:
            mask = torch.log(valid_mask.view(valid_mask.size()[0], 1, 1,
                valid_mask.size()[1]))
            scaled_scores += mask
        attention_weights = self.softmax(scaled_scores)
        return torch.matmul(attention_weights, tiled_inputs)

    def _concat_heads(self, outputs):
        max_contexts = outputs.size()[2]
        tensor = outputs.transpose(1, 2)
        return tensor.contiguous().view([-1, max_contexts, self.model_dim *
            self.num_heads])


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'num_heads': 4, 'model_dim': 4, 'dropout_keep_prob': 0.5}]
