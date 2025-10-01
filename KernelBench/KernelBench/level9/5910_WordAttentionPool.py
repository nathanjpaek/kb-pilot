from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class WordAttentionPool(nn.Module):

    def __init__(self, cfg):
        super(WordAttentionPool, self).__init__()
        input_size = cfg.INPUT_SIZE
        hidden_size = cfg.HIDDEN_SIZE
        self.stride = cfg.STRIDE
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.text_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, visual_input, text_feature):
        _, _, v_len = visual_input.shape
        vis_att = torch.relu(self.vis_conv(visual_input))
        text_att = torch.relu(self.text_linear(text_feature))
        att = torch.matmul(text_att.unsqueeze(1), vis_att).transpose(1, 2)
        seg_list = []
        for i in range(v_len // self.stride):
            vis_seg = visual_input[:, :, self.stride * i:self.stride * (i + 1)
                ].transpose(1, 2)
            att_seg = torch.softmax(att[:, self.stride * i:self.stride * (i +
                1), :], dim=1)
            vis_new = torch.sum(vis_seg * att_seg, dim=1)
            seg_list.append(vis_new)
        vis_out = torch.relu(self.vis_conv(torch.stack(seg_list, dim=2)))
        return vis_out


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'cfg': _mock_config(INPUT_SIZE=4, HIDDEN_SIZE=4, STRIDE=4)}]
