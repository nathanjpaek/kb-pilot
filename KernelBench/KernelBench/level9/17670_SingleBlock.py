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


class OneSideInterModalityUpdate(nn.Module):
    """
    one-side Inter-Modality Attention Flow
    according to the paper, instead of parallel V->Q & Q->V, we first to V->Q and then Q->V
    """

    def __init__(self, src_size, tgt_size, output_size, num_head, drop=0.0):
        super(OneSideInterModalityUpdate, self).__init__()
        self.src_size = src_size
        self.tgt_size = tgt_size
        self.output_size = output_size
        self.num_head = num_head
        self.src_lin = FCNet(src_size, output_size * 2, drop=drop, activate
            ='relu')
        self.tgt_lin = FCNet(tgt_size, output_size, drop=drop, activate='relu')
        self.tgt_output = FCNet(output_size + tgt_size, output_size, drop=
            drop, activate='relu')

    def forward(self, src, tgt):
        """
        :param src: eeg feature [batch, regions, feature_size]
        :param tgt: eye feature [batch, regions, feature_size]
        :return:
        """
        _batch_size, _num_src = src.shape[0], src.shape[1]
        tgt.shape[1]
        src_tran = self.src_lin(src)
        tgt_tran = self.tgt_lin(tgt)
        src_key, src_val = torch.split(src_tran, src_tran.size(2) // 2, dim=2)
        tgt_query = tgt_tran
        src_key_set = torch.split(src_key, src_key.size(2) // self.num_head,
            dim=2)
        src_val_set = torch.split(src_val, src_val.size(2) // self.num_head,
            dim=2)
        tgt_query_set = torch.split(tgt_query, tgt_query.size(2) // self.
            num_head, dim=2)
        for i in range(self.num_head):
            src_key_slice, tgt_query_slice, src_val_slice = src_key_set[i
                ], tgt_query_set[i], src_val_set[i]
            src2tgt = tgt_query_slice @ src_key_slice.transpose(1, 2) / (self
                .output_size // self.num_head) ** 0.5
            interMAF_src2tgt = F.softmax(src2tgt, dim=2).unsqueeze(3)
            tgt_update = (interMAF_src2tgt * src_val_slice.unsqueeze(1)).sum(2
                ) if i == 0 else torch.cat((tgt_update, (interMAF_src2tgt *
                src_val_slice.unsqueeze(1)).sum(2)), dim=2)
        cat_tgt = torch.cat((tgt, tgt_update), dim=2)
        tgt_updated = self.tgt_output(cat_tgt)
        return tgt_updated


class DyIntraModalityUpdate(nn.Module):
    """
    Dynamic Intra-Modality Attention Flow
    """

    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):
        super(DyIntraModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head
        self.v4q_gate_lin = FCNet(v_size, output_size, drop=drop)
        self.q4v_gate_lin = FCNet(q_size, output_size, drop=drop)
        self.v_lin = FCNet(v_size, output_size * 3, drop=drop, activate='relu')
        self.q_lin = FCNet(q_size, output_size * 3, drop=drop, activate='relu')
        self.v_output = FCNet(output_size, output_size, drop=drop, activate
            ='relu')
        self.q_output = FCNet(output_size, output_size, drop=drop, activate
            ='relu')
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, v, q):
        """
        :param v: [batch_size, num_obj, feature_size]
        :param q: [batch_size, max_len, feature_size]

        :return:
        """
        _batch_size, num_obj = v.shape[0], v.shape[1]
        max_len = q.shape[1]
        v_mean = v.sum(1) / num_obj
        q_mean = q.sum(1) / max_len
        v4q_gate = self.sigmoid(self.v4q_gate_lin(v_mean)).unsqueeze(1)
        q4v_gate = self.sigmoid(self.q4v_gate_lin(q_mean)).unsqueeze(1)
        v_tran = self.v_lin(v)
        q_tran = self.q_lin(q)
        v_key, v_query, v_val = torch.split(v_tran, v_tran.size(2) // 3, dim=2)
        q_key, q_query, q_val = torch.split(q_tran, q_tran.size(2) // 3, dim=2)
        gated_v_query = (1 + q4v_gate) * v_query
        gated_v_key = (1 + q4v_gate) * v_key
        gated_v_val = (1 + q4v_gate) * v_val
        gated_q_query = (1 + v4q_gate) * q_query
        gated_q_key = (1 + v4q_gate) * q_key
        gated_q_val = (1 + v4q_gate) * q_val
        v_key_set = torch.split(gated_v_key, gated_v_key.size(2) // self.
            num_head, dim=2)
        v_query_set = torch.split(gated_v_query, gated_v_query.size(2) //
            self.num_head, dim=2)
        v_val_set = torch.split(gated_v_val, gated_v_val.size(2) // self.
            num_head, dim=2)
        q_key_set = torch.split(gated_q_key, gated_q_key.size(2) // self.
            num_head, dim=2)
        q_query_set = torch.split(gated_q_query, gated_q_query.size(2) //
            self.num_head, dim=2)
        q_val_set = torch.split(gated_q_val, gated_q_val.size(2) // self.
            num_head, dim=2)
        for i in range(self.num_head):
            v_key_slice, v_query_slice, v_val_slice = v_key_set[i
                ], v_query_set[i], v_val_set[i]
            q_key_slice, q_query_slice, q_val_slice = q_key_set[i
                ], q_query_set[i], q_val_set[i]
            v2v = v_query_slice @ v_key_slice.transpose(1, 2) / (self.
                output_size // self.num_head) ** 0.5
            q2q = q_query_slice @ q_key_slice.transpose(1, 2) / (self.
                output_size // self.num_head) ** 0.5
            dyIntranMAF_v2v = F.softmax(v2v, dim=2).unsqueeze(3)
            dyIntranMAF_q2q = F.softmax(q2q, dim=2).unsqueeze(3)
            v_update = (dyIntranMAF_v2v * v_val_slice.unsqueeze(1)).sum(2
                ) if i == 0 else torch.cat((v_update, (dyIntranMAF_v2v *
                v_val_slice.unsqueeze(1)).sum(2)), dim=2)
            q_update = (dyIntranMAF_q2q * q_val_slice.unsqueeze(1)).sum(2
                ) if i == 0 else torch.cat((q_update, (dyIntranMAF_q2q *
                q_val_slice.unsqueeze(1)).sum(2)), dim=2)
        updated_v = self.v_output(v + v_update)
        updated_q = self.q_output(q + q_update)
        return updated_v, updated_q


class SingleBlock(nn.Module):
    """
        Single Block Inter- and Intra modality stack multiple times, in such circumstance, all the
        basic blocks share the same parameters in the model
    """

    def __init__(self, num_blocks, v_size, q_size, output_size,
        num_inter_head, num_intra_head, drop=0.0):
        super(SingleBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_inter_head = num_inter_head
        self.num_intra_head = num_intra_head
        self.num_block = num_blocks
        self.v_lin = FCNet(v_size, output_size, drop=drop, activate='relu')
        self.q_lin = FCNet(q_size, output_size, drop=drop, activate='relu')
        self.v2q_interBlock = OneSideInterModalityUpdate(output_size,
            output_size, output_size, num_inter_head, drop)
        self.q2v_interBlock = OneSideInterModalityUpdate(output_size,
            output_size, output_size, num_inter_head, drop)
        self.intraBlock = DyIntraModalityUpdate(output_size, output_size,
            output_size, num_intra_head, drop)

    def forward(self, v, q):
        """
        :param v: eeg feature [batch_size, regions, feature_size]
        :param q: eye feature [batch_size, regions, feature_size]
        :return:
        """
        v = self.v_lin(v)
        q = self.q_lin(q)
        v_container = [v]
        q_container = [q]
        result_v = [v]
        result_q = [q]
        for i in range(self.num_block):
            q1 = self.v2q_interBlock(v_container[-1], q_container[-1])
            q_container.append(q1)
            v1 = self.q2v_interBlock(q_container[-1], v_container[-1])
            v_container.append(v1)
            v2, q2 = self.intraBlock(v_container[-1] + v_container[-2], 
                q_container[-1] + q_container[-2])
            v_container.append(v2)
            q_container.append(q2)
            result_v.append(v1)
            result_v.append(v2)
            result_q.append(q1)
            result_q.append(q2)
            v_container.append(v_container[-1] + v_container[-2] +
                v_container[-3])
            q_container.append(q_container[-1] + q_container[-2] +
                q_container[-3])
        return sum(result_v), sum(result_q)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'num_blocks': 4, 'v_size': 4, 'q_size': 4, 'output_size': 
        4, 'num_inter_head': 4, 'num_intra_head': 4}]
