import torch
import torch.nn.functional as F


class Attention(torch.nn.Module):
    """Scaled dot product attention."""

    def __init__(self, hidden_dim, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.projection_layer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, atten_post):
        posts_attention_values = self.projection_layer(atten_post)
        posts_attention_weights = F.softmax(posts_attention_values.permute(
            0, 2, 1), dim=-1)
        del posts_attention_values
        torch.cuda.empty_cache()
        self_atten_output_post = torch.matmul(posts_attention_weights,
            atten_post).squeeze(dim=1)
        return self_atten_output_post, posts_attention_weights


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4}]
