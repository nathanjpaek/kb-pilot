from _paritybench_helpers import _mock_config
import time
import torch
from torch import nn


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def forward(self, *input):
        raise NotImplementedError


class MFBFusion(BaseModel):

    def __init__(self, mfb_fusion_option: 'MFBFusionOption'):
        super(MFBFusion, self).__init__()
        self.out_dim = mfb_fusion_option.outdim
        self.linear1 = nn.Linear(mfb_fusion_option.text_size,
            mfb_fusion_option.joint_emb_size)
        self.linear2 = nn.Linear(mfb_fusion_option.image_size,
            mfb_fusion_option.joint_emb_size)
        self.dropout = nn.Dropout(p=mfb_fusion_option.dropout)

    def forward(self, text_feat, image_feat):
        batch_size = text_feat.size(0)
        text_proj = self.linear1(text_feat)
        image_proj = self.linear2(image_feat)
        mm_eltwise = torch.mul(text_proj, image_proj)
        mm_drop = self.dropout(mm_eltwise)
        mm_resh = mm_drop.view(batch_size, 1, self.out_dim, -1)
        mm_sumpool = torch.sum(mm_resh, 3, keepdim=True)
        mfb_out = torch.squeeze(mm_sumpool)
        return mfb_out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'mfb_fusion_option': _mock_config(outdim=4, text_size=4,
        joint_emb_size=4, image_size=4, dropout=0.5)}]
