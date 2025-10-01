import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class InterModalityUpdate(nn.Module):
    """
    Inter-modality Attention Flow
    """

    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):
        super(InterModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head
        self.v_lin = nn.Linear(v_size, output_size * 3)
        self.q_lin = nn.Linear(q_size, output_size * 3)
        self.v_output = nn.Linear(output_size + v_size, output_size)
        self.q_output = nn.Linear(output_size + q_size, output_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        batch_size, num_obj = v_mask.shape
        _, max_len = q_mask.shape
        v_trans = self.v_lin(self.drop(self.relu(v)))
        q_trans = self.q_lin(self.drop(self.relu(q)))
        v_trans = v_trans * v_mask.unsqueeze(2)
        q_trans = q_trans * q_mask.unsqueeze(2)
        v_k, v_q, v_v = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_k, q_q, q_v = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
        vk_set = torch.split(v_k, v_k.size(2) // self.num_head, dim=2)
        vq_set = torch.split(v_q, v_q.size(2) // self.num_head, dim=2)
        vv_set = torch.split(v_v, v_v.size(2) // self.num_head, dim=2)
        qk_set = torch.split(q_k, q_k.size(2) // self.num_head, dim=2)
        qq_set = torch.split(q_q, q_q.size(2) // self.num_head, dim=2)
        qv_set = torch.split(q_v, q_v.size(2) // self.num_head, dim=2)
        for i in range(self.num_head):
            vk_slice, vq_slice, vv_slice = vk_set[i], vq_set[i], vv_set[i]
            qk_slice, qq_slice, qv_slice = qk_set[i], qq_set[i], qv_set[i]
            q2v = (vq_slice @ qk_slice.transpose(1, 2)).masked_fill(q_mask.
                unsqueeze(1).expand([batch_size, num_obj, max_len]) == 0, -
                1000000000.0) / (self.output_size // self.num_head) ** 0.5
            v2q = (qq_slice @ vk_slice.transpose(1, 2)).masked_fill(v_mask.
                unsqueeze(1).expand([batch_size, max_len, num_obj]) == 0, -
                1000000000.0) / (self.output_size // self.num_head) ** 0.5
            interMAF_q2v = F.softmax(q2v, dim=2)
            interMAF_v2q = F.softmax(v2q, dim=2)
            v_update = interMAF_q2v @ qv_slice if i == 0 else torch.cat((
                v_update, interMAF_q2v @ qv_slice), dim=2)
            q_update = interMAF_v2q @ vv_slice if i == 0 else torch.cat((
                q_update, interMAF_v2q @ vv_slice), dim=2)
        cat_v = torch.cat((v, v_update), dim=2)
        cat_q = torch.cat((q, q_update), dim=2)
        updated_v = self.v_output(self.drop(cat_v))
        updated_q = self.q_output(self.drop(cat_q))
        return updated_v, updated_q


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4]
        ), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'v_size': 4, 'q_size': 4, 'output_size': 4, 'num_head': 4}]
