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


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'src_size': 4, 'tgt_size': 4, 'output_size': 4, 'num_head': 4}
        ]
