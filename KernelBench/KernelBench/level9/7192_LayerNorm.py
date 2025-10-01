import torch


class LayerNorm(torch.nn.Module):
    """
    A vanilla implementation of layer normalization  https://arxiv.org/pdf/1607.06450.pdf
    norm_x = (x - mean) / sqrt((x - mean) ^ 2)
    This does not include the trainable parameters gamma and beta for performance speed.
    Typically, this is norm_x * gamma + beta
    """

    def forward(self, layer_activations: 'torch.Tensor') ->torch.Tensor:
        mean = torch.mean(layer_activations, dim=-1, keepdim=True)
        var = torch.mean((layer_activations - mean) ** 2, dim=-1, keepdim=True)
        return (layer_activations - mean) / torch.sqrt(var + 1e-05)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
