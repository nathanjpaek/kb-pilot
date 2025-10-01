import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNet(nn.Module):

    def __init__(self, in_size, out_size, activate=None, drop=0.0):
        super(FCNet, self).__init__()
        self.lin = nn.Linear(in_size, out_size)
        self.drop_value = drop
        self.drop = nn.Dropout(drop)
        self.activate = activate.lower() if activate is not None else None
        if activate == 'relu':
            self.ac_fn = nn.ReLU()
        elif activate == 'sigmoid':
            self.ac_fn = nn.Sigmoid()
        elif activate == 'tanh':
            self.ac_fn = nn.Tanh()

    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)
        x = self.lin(x)
        if self.activate is not None:
            x = self.ac_fn(x)
        return x


class InterModalityUpdate(nn.Module):
    """
    Inter-Modality Attention Flow
    """

    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):
        super(InterModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head
        self.v_lin = FCNet(v_size, output_size * 3, drop=drop, activate='relu')
        self.q_lin = FCNet(q_size, output_size * 3, drop=drop, activate='relu')
        self.v_output = FCNet(output_size + v_size, output_size, drop=drop,
            activate='relu')
        self.q_output = FCNet(output_size + q_size, output_size, drop=drop,
            activate='relu')

    def forward(self, v, q):
        """
        :param v: eeg feature [batch, regions, feature_size]
        :param q: eye feature [batch, regions, feature_size]
        :return:
        """
        _batch_size, _num_obj = v.shape[0], v.shape[1]
        q.shape[1]
        v_tran = self.v_lin(v)
        q_tran = self.q_lin(q)
        v_key, v_query, v_val = torch.split(v_tran, v_tran.size(2) // 3, dim=2)
        q_key, q_query, q_val = torch.split(q_tran, q_tran.size(2) // 3, dim=2)
        v_key_set = torch.split(v_key, v_key.size(2) // self.num_head, dim=2)
        v_query_set = torch.split(v_query, v_query.size(2) // self.num_head,
            dim=2)
        v_val_set = torch.split(v_val, v_val.size(2) // self.num_head, dim=2)
        q_key_set = torch.split(q_key, q_key.size(2) // self.num_head, dim=2)
        q_query_set = torch.split(q_query, q_query.size(2) // self.num_head,
            dim=2)
        q_val_set = torch.split(q_val, q_val.size(2) // self.num_head, dim=2)
        for i in range(self.num_head):
            v_key_slice, v_query_slice, v_val_slice = v_key_set[i
                ], v_query_set[i], v_val_set[i]
            q_key_slice, q_query_slice, q_val_slice = q_key_set[i
                ], q_query_set[i], q_val_set[i]
            q2v = v_query_slice @ q_key_slice.transpose(1, 2) / (self.
                output_size // self.num_head) ** 0.5
            v2q = q_query_slice @ v_key_slice.transpose(1, 2) / (self.
                output_size // self.num_head) ** 0.5
            interMAF_q2v = F.softmax(q2v, dim=2).unsqueeze(3)
            interMAF_v2q = F.softmax(v2q, dim=2).unsqueeze(3)
            v_update = (interMAF_q2v * q_val_slice.unsqueeze(1)).sum(2
                ) if i == 0 else torch.cat((v_update, (interMAF_q2v *
                q_val_slice.unsqueeze(1)).sum(2)), dim=2)
            q_update = (interMAF_v2q * v_val_slice.unsqueeze(1)).sum(2
                ) if i == 0 else torch.cat((q_update, (interMAF_v2q *
                v_val_slice.unsqueeze(1)).sum(2)), dim=2)
        cat_v = torch.cat((v, v_update), dim=2)
        cat_q = torch.cat((q, q_update), dim=2)
        updated_v = self.v_output(cat_v)
        updated_q = self.q_output(cat_q)
        return updated_v, updated_q


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'v_size': 4, 'q_size': 4, 'output_size': 4, 'num_head': 4}]
