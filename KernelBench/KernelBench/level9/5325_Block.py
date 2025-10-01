import torch
import torch.nn as nn


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.
        device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head self attention layer
    
    """

    def __init__(self, in_features, num_heads=8, qkv_bias=False,
        attention_drop=0.0, proj_drop=0.0):
        """
        Args:
            in_features (int): input dimension
            num_heads (int, optional): [description]. Defaults to 8.
            qkv_bias (bool, optional): [description]. Defaults to False.
            attention_drop ([type], optional): [description]. Defaults to 0..
            proj_drop ([type], optional): [description]. Defaults to 0..
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        head_dims = in_features // num_heads
        self.scale = head_dims ** -0.5
        self.qkv = nn.Linear(in_features, in_features * 3, bias=qkv_bias)
        self.attention_drop = nn.Dropout(attention_drop)
        self.projection = nn.Linear(in_features, in_features)
        self.projection_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """for iterating self attention, output shape must be equal to input shape"""
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        query, key, value = qkv
        attn_out = torch.matmul(query, key.transpose(-2, -1))
        attn_out *= self.scale
        attn_out = torch.softmax(attn_out, dim=-1)
        attn_out = self.attention_drop(attn_out)
        attn_out = torch.matmul(attn_out, value)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        out = self.projection(attn_out)
        out = self.projection_drop(out)
        return out


class MLP(nn.Module):
    """
    Multi Layer Perceptron
    I do compose it of two fully connected layers(a.k.a Linear layer)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
        activation_layer=nn.GELU, drop_rate=0.0):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = activation_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    """
    Block is composed of multi-head attention & MLP(feedforward).
        (1) norm_layer 
        (2) multi-head attention 
        (3) shortcut 
        (4) norm_layer 
        (5) MLP 
        (6) shortcut
    It will be iterated several times
    """

    def __init__(self, in_features, num_heads, mlp_ratio=4.0, qkv_bias=
        False, drop_rate=0.0, attn_drop_rate=0.0, drop_path=0.0,
        activation_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        Args:
            in_features (int): input dimension
            num_heads (int): number of heads to use
            mlp_ratio (float, optional): hidden dimension size of MLP layer. Defaults to 4..
            qkv_bias (bool, optional): if using qkv hidden layer's bias. Defaults to False.
            drop_rate (float, optional): dropout ratio. Defaults to 0..
            attn_drop_rate (float, optional): dropout ratio in multi-head attention. Defaults to 0..
            drop_path (float, optional): ???. Defaults to 0..
            activation_layer (nn.Module, optional): activation function(layer). Defaults to nn.GELU.
            norm_layer (nn.Module, optional): normalization layer. Defaults to nn.LayerNorm.
        """
        super(Block, self).__init__()
        self.norm1 = norm_layer(in_features)
        self.multihead_attention = MultiHeadAttention(in_features,
            num_heads=num_heads, qkv_bias=qkv_bias, attention_drop=
            attn_drop_rate, proj_drop=drop_rate)
        self.drop_path = DropPath(drop_prob=drop_path
            ) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(in_features)
        mlp_hidden_features = int(in_features * mlp_ratio)
        self.mlp = MLP(in_features, hidden_features=mlp_hidden_features,
            activation_layer=activation_layer, drop_rate=drop_rate)

    def forward(self, x_in):
        x = self.norm1(x_in)
        x_in = x_in + self.drop_path(self.multihead_attention(x))
        x = self.norm2(x_in)
        x = x_in + self.drop_path(self.mlp(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'num_heads': 4}]
