import torch
import torch.nn as nn


def init_weights(m):
    None
    if type(m) == nn.Linear:
        None
    else:
        None


class MMTM(nn.Module):

    def __init__(self, dim_visual, dim_skeleton, ratio):
        super(MMTM, self).__init__()
        dim = dim_visual + dim_skeleton
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)
        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        with torch.no_grad():
            self.fc_squeeze.apply(init_weights)
            self.fc_visual.apply(init_weights)
            self.fc_skeleton.apply(init_weights)

    def forward(self, visual, skeleton):
        squeeze_array = []
        for tensor in [visual, skeleton]:
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_array.append(torch.mean(tview, dim=-1))
        squeeze = torch.cat(squeeze_array, 1)
        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)
        vis_out = self.fc_visual(excitation)
        sk_out = self.fc_skeleton(excitation)
        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)
        dim_diff = len(visual.shape) - len(vis_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)
        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)
        return visual * vis_out, skeleton * sk_out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_visual': 4, 'dim_skeleton': 4, 'ratio': 4}]
