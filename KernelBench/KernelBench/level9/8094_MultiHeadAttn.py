import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttn(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, num_head,
        dropatt=0.0, act_func='softmax', add_zero_attn=False, pre_lnorm=
        False, post_lnorm=False):
        super(MultiHeadAttn, self).__init__()
        assert hidden_dim % num_head == 0
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.dropatt = nn.Dropout(dropatt)
        head_dim = hidden_dim // num_head
        self.q_net = nn.Linear(query_dim, hidden_dim, bias=False)
        self.k_net = nn.Linear(key_dim, hidden_dim, bias=False)
        self.v_net = nn.Linear(value_dim, hidden_dim, bias=False)
        self.o_net = nn.Linear(hidden_dim, query_dim, bias=False)
        self.scale = 1 / head_dim ** 0.5
        self.act_func = act_func
        self.add_zero_attn = add_zero_attn
        self.pre_lnorm = pre_lnorm
        self.post_lnorm = post_lnorm
        if pre_lnorm:
            self.q_layer_norm = nn.LayerNorm(query_dim)
            self.k_layer_norm = nn.LayerNorm(key_dim)
            self.v_layer_norm = nn.LayerNorm(value_dim)
        if post_lnorm:
            self.o_layer_norm = nn.LayerNorm(query_dim)
        for net in [self.q_net, self.k_net, self.v_net, self.o_net]:
            nn.init.xavier_uniform_(net.weight, 1.0)
            if hasattr(net, 'bias') and net.bias is not None:
                nn.init.constant_(net.bias, 0.0)
        if self.pre_lnorm:
            for layer_norm in [self.q_layer_norm, self.k_layer_norm, self.
                v_layer_norm]:
                if hasattr(layer_norm, 'weight'):
                    nn.init.normal_(layer_norm.weight, 1.0, self.scale)
                if hasattr(layer_norm, 'bias') and layer_norm.bias is not None:
                    nn.init.constant_(layer_norm.bias, 0.0)
        if self.post_lnorm:
            if hasattr(self.o_layer_norm, 'weight'):
                nn.init.normal_(self.o_layer_norm.weight, 1.0, self.scale)
            if hasattr(self.o_layer_norm, 'bias'
                ) and self.o_layer_norm.bias is not None:
                nn.init.constant_(self.o_layer_norm.bias, 0.0)

    def forward(self, query, key, value, attn_mask=None):
        bsz = query.size(0)
        if self.add_zero_attn:
            key = torch.cat([key, torch.zeros((bsz, 1) + key.size()[2:],
                dtype=key.dtype, device=key.device)], dim=1)
            value = torch.cat([value, torch.zeros((bsz, 1) + value.size()[2
                :], dtype=value.dtype, device=value.device)], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.ones((bsz, 1),
                    dtype=attn_mask.dtype, device=attn_mask.device)], dim=1)
        qlen, klen, vlen = query.size(1), key.size(1), value.size(1)
        if self.pre_lnorm:
            query = self.q_layer_norm(query)
            key = self.k_layer_norm(key)
            value = self.v_layer_norm(value)
        head_q = self.q_net(query).view(bsz, qlen, self.num_head, self.
            hidden_dim // self.num_head)
        head_k = self.k_net(key).view(bsz, klen, self.num_head, self.
            hidden_dim // self.num_head)
        head_v = self.v_net(value).view(bsz, vlen, self.num_head, self.
            hidden_dim // self.num_head)
        attn_score = torch.einsum('bind,bjnd->bijn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_((attn_mask == 0).unsqueeze(1).
                    unsqueeze(-1), _INF)
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_((attn_mask == 0).unsqueeze(-1), _INF)
        if self.act_func is None or self.act_func == 'None':
            attn_prob = attn_score
        elif self.act_func == 'softmax':
            attn_prob = F.softmax(attn_score, dim=2)
        elif self.act_func == 'sigmoid':
            attn_prob = F.sigmoid(attn_score)
        elif self.act_func == 'tanh':
            attn_prob = F.tanh(attn_score)
        elif self.act_func == 'relu':
            attn_prob = F.relu(attn_score)
        elif self.act_func == 'leaky_relu':
            attn_prob = F.leaky_relu(attn_score)
        elif self.act_func == 'maximum':
            max_score = torch.max(attn_score, dim=2, keepdim=True)[0]
            max_mask = attn_score == max_score
            cnt = torch.sum(max_mask, dim=2, keepdim=True)
            attn_prob = max_mask.float() / cnt.float()
        else:
            raise NotImplementedError
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum('bijn,bjnd->bind', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(bsz, qlen, self.hidden_dim)
        attn_out = self.o_net(attn_vec)
        if self.post_lnorm:
            attn_out = self.o_layer_norm(attn_out)
        return attn_out

    def get_output_dim(self):
        return self.query_dim


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'query_dim': 4, 'key_dim': 4, 'value_dim': 4, 'hidden_dim':
        4, 'num_head': 4}]
