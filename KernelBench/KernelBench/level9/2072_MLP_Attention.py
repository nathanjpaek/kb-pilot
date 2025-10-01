import torch
import torch.nn as nn
import torch.nn.init as init


class MLP_Attention(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(MLP_Attention, self).__init__()
        self.linear_X = nn.Linear(input_size, hidden_size, bias=True)
        self.linear_ref = nn.Linear(input_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, out_features=1)

    def init_weight(self):
        init.xavier_normal_(self.linear_X.weight)
        init.xavier_normal_(self.linear_ref.weight)
        init.xavier_normal_(self.v.weight)
        init.constant_(self.linear1.bias, 0.0)
        init.constant_(self.linear2.bias, 0.0)
        init.constant_(self.v.bias, 0.0)

    def forward(self, X, ref):
        batch_size, n_X, _ = X.shape
        _, n_ref, _ = ref.shape
        stacking_X = self.linear_X(X).view(batch_size, n_X, 1, -1).repeat(1,
            1, n_ref, 1)
        stacking_ref = self.linear_ref(ref).view(batch_size, 1, n_ref, -1
            ).repeat(1, n_X, 1, 1)
        out = self.v(torch.tanh(stacking_X + stacking_ref)).squeeze()
        attention_scores = torch.softmax(out, dim=1)
        weighted_X = torch.einsum('bxe,bxr->bre', X, attention_scores)
        return weighted_X


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
