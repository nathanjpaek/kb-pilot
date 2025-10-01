import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class DyIntraModalityUpdate(nn.Module):
    """
    Dynamic Intra-modality Attention Flow
    """

    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):
        super(DyIntraModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head
        self.v4q_gate_lin = nn.Linear(v_size, output_size)
        self.q4v_gate_lin = nn.Linear(q_size, output_size)
        self.v_lin = nn.Linear(v_size, output_size * 3)
        self.q_lin = nn.Linear(q_size, output_size * 3)
        self.v_output = nn.Linear(output_size, output_size)
        self.q_output = nn.Linear(output_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
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
        v_mean = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
        q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)
        v4q_gate = self.sigmoid(self.v4q_gate_lin(self.drop(self.relu(v_mean)))
            ).unsqueeze(1)
        q4v_gate = self.sigmoid(self.q4v_gate_lin(self.drop(self.relu(q_mean)))
            ).unsqueeze(1)
        v_trans = self.v_lin(self.drop(self.relu(v)))
        q_trans = self.q_lin(self.drop(self.relu(q)))
        v_trans = v_trans * v_mask.unsqueeze(2)
        q_trans = q_trans * q_mask.unsqueeze(2)
        v_k, v_q, v_v = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_k, q_q, q_v = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
        new_vq = (1 + q4v_gate) * v_q
        new_vk = (1 + q4v_gate) * v_k
        new_qq = (1 + v4q_gate) * q_q
        new_qk = (1 + v4q_gate) * q_k
        vk_set = torch.split(new_vk, new_vk.size(2) // self.num_head, dim=2)
        vq_set = torch.split(new_vq, new_vq.size(2) // self.num_head, dim=2)
        vv_set = torch.split(v_v, v_v.size(2) // self.num_head, dim=2)
        qk_set = torch.split(new_qk, new_qk.size(2) // self.num_head, dim=2)
        qq_set = torch.split(new_qq, new_qq.size(2) // self.num_head, dim=2)
        qv_set = torch.split(q_v, q_v.size(2) // self.num_head, dim=2)
        for i in range(self.num_head):
            vk_slice, vq_slice, vv_slice = vk_set[i], vq_set[i], vv_set[i]
            qk_slice, qq_slice, qv_slice = qk_set[i], qq_set[i], qv_set[i]
            v2v = (vq_slice @ vk_slice.transpose(1, 2)).masked_fill(v_mask.
                unsqueeze(1).expand([batch_size, num_obj, num_obj]) == 0, -
                1000000000.0) / (self.output_size // self.num_head) ** 0.5
            q2q = (qq_slice @ qk_slice.transpose(1, 2)).masked_fill(q_mask.
                unsqueeze(1).expand([batch_size, max_len, max_len]) == 0, -
                1000000000.0) / (self.output_size // self.num_head) ** 0.5
            dyIntraMAF_v2v = F.softmax(v2v, dim=2)
            dyIntraMAF_q2q = F.softmax(q2q, dim=2)
            v_update = dyIntraMAF_v2v @ vv_slice if i == 0 else torch.cat((
                v_update, dyIntraMAF_v2v @ vv_slice), dim=2)
            q_update = dyIntraMAF_q2q @ qv_slice if i == 0 else torch.cat((
                q_update, dyIntraMAF_q2q @ qv_slice), dim=2)
        updated_v = self.v_output(self.drop(v + v_update))
        updated_q = self.q_output(self.drop(q + q_update))
        return updated_v, updated_q


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4]
        ), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'v_size': 4, 'q_size': 4, 'output_size': 4, 'num_head': 4}]
