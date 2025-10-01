import torch
import torch.nn as nn
from abc import ABCMeta
from torch.utils import model_zoo


class BaseModule(nn.Module, metaclass=ABCMeta):

    @classmethod
    def load(cls, config, state_dict=None):
        model = cls.from_cfg(config)
        if model is not None and state_dict is not None:
            model.load_state_dict(state_dict)
        return model

    @classmethod
    def from_cfg(cls, config):
        raise NotImplementedError

    @staticmethod
    def parse_pytorch_file(file_path):
        if urlparse(file_path).scheme in ('http', 'https'):
            state_dict = model_zoo.load_url(file_path, progress=True)
        else:
            assert osp.exists(file_path), f'File does not exist: {file_path}'
            None
            state_dict = torch.load(file_path, map_location=torch.device('cpu')
                )
        return state_dict


class LocalFeatureEncoder(BaseModule):

    def __init__(self, num_joints, z_dim, point_feature_len):
        super().__init__()
        self.num_joints = num_joints
        self.z_dim = z_dim
        self.point_feature_len = point_feature_len
        self.net = nn.Conv1d(in_channels=self.num_joints * self.z_dim,
            out_channels=self.num_joints * self.point_feature_len,
            kernel_size=(1,), groups=self.num_joints)

    def get_out_dim(self):
        return self.point_feature_len

    @classmethod
    def from_cfg(cls, config):
        return cls(num_joints=config['num_joints'], z_dim=config['z_dim'],
            point_feature_len=config['point_feature_len'])

    def forward(self, shape_code, structure_code, pose_code, lbs_weights):
        """
            skinning_weights: B x T x K
        """
        B, T, K = lbs_weights.shape
        assert K == self.num_joints
        global_feature = []
        if shape_code is not None:
            global_feature.append(shape_code)
        if structure_code is not None:
            global_feature.append(structure_code)
        if pose_code is not None:
            global_feature.append(pose_code)
        global_feature = torch.cat(global_feature, dim=-1)
        global_feature = global_feature.unsqueeze(1).repeat(1, K, 1)
        global_feature = global_feature.view(B, -1, 1)
        local_feature = self.net(global_feature).view(B, K, -1)
        local_feature = local_feature.unsqueeze(1).repeat(1, T, 1, 1)
        local_feature = local_feature.view(B * T, K, -1)
        lbs_weights = lbs_weights.view(B * T, 1, K)
        local_feature = torch.bmm(lbs_weights, local_feature).view(B, T, -1)
        return local_feature


def get_inputs():
    return [torch.rand([4, 1]), torch.rand([4, 1]), torch.rand([4, 2]),
        torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'num_joints': 4, 'z_dim': 4, 'point_feature_len': 4}]
