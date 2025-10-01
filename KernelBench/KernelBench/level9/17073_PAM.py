import torch
from typing import Union
import torch.nn.functional as F
from typing import Tuple
from torch import nn
from typing import Dict
from typing import List


def strip_param_name(param_name: 'str') ->str:
    """Input an module's param name, return it's origin name with out parent modules' name

    Args:
        param_name (str): parameter name with it's parent module prefix

    Returns:
        Param's origin name
    """
    splits = param_name.rsplit('.', maxsplit=2)
    return '.'.join(splits[-2:])


def strip_named_params(named_params: 'kParamDictType') ->Union[Dict[str, nn
    .Parameter], Dict[str, torch.Tensor]]:
    """Strip all param names in modules' state_dict

    Args:
        named_params: module's state_dict

    Returns:
        A Dict with key of stripped param name and it's origin value.

    See Also:
        strip_param_name
    """
    return {strip_param_name(k): v for k, v in named_params.items()}


def update_model_state_dict(model: 'nn.Module', weight_dict:
    'kParamDictType', verbosity: 'int'=2):
    """Update model's state_dict with pretrain_dict

        Args:
            model: model to be updated
            weight_dict: pretrain state dict
            verbosity: 0: No info; 1: Show weight loaded; 2: Show weight missed; 3: Show weight Redundant
    """
    model_dict = model.state_dict()
    update_dict = {k: weight_dict[k] for k in model_dict.keys() if k in
        weight_dict}
    model_dict.update(update_dict)
    model.load_state_dict(model_dict, strict=True)
    if verbosity >= 1:
        for k in sorted(update_dict.keys()):
            None
    if verbosity >= 2:
        missing_dict = {k: model_dict[k] for k in model_dict.keys() if k not in
            weight_dict}
        for k in sorted(missing_dict.keys()):
            None
    if verbosity >= 3:
        redundant_dict = {k: weight_dict[k] for k in weight_dict.keys() if 
            k not in model_dict}
        for k in sorted(redundant_dict.keys()):
            None


class Network(nn.Module):
    """Torch base networks with default I/O info and initialization"""
    in_batch_size = None
    out_batch_size = None
    in_channel = None
    out_channel = None
    in_height = None
    out_height = None
    in_width = None
    out_width = None

    def initialize(self, pretrain_dict: 'kParamDictType', verbosity: 'int'=
        2, **kwargs):
        """Default initialize. Initialize every sub module with type Network using pretrain_dict

        Args:
            pretrain_dict: pretrain state dict
            verbosity: 0: No info; 1: Show weight loaded; 2: Show weight missed; 3: Show weight Redundant. (Default: 2)
        """
        for child in self.children():
            if isinstance(child, Network):
                child.initialize(pretrain_dict, verbosity, **kwargs)

    def find_match_miss(self, pretrain_dict: 'kParamDictType') ->Tuple[
        Union[Dict[str, nn.Parameter], Dict[str, torch.Tensor]], List[str]]:
        """Find params in pretrain dict with the same suffix name of params of model.

        Args:
            pretrain_dict: pretrain dict with key: parameter

        Returns:
            matched_dict: keys with corresponding pretrain values of model's state_dict whose suffix name can be found
                in pretrain_dict
            missed_keys: keys of model's state_dict whose suffix name can not be found in pretrain_dict
        """
        return find_match_miss(self, pretrain_dict)

    def update_model_state_dict(self, weight_dict: 'kParamDictType',
        verbosity: 'int'=2):
        """Update model's state_dict with pretrain_dict

        Args:
            weight_dict: pretrain state dict
            verbosity: 0: No info; 1: Show weight loaded; 2: Show weight missed; 3: Show weight Redundant
        """
        update_model_state_dict(self, weight_dict, verbosity=verbosity)

    @property
    def strip_named_params(self):
        return strip_named_params(self.state_dict())


class PAM(Network):

    def __init__(self, in_channel, mid_channel, num_classes, layer_idx,
        branch_idx):
        """init function for positional attention module
        
        Parameters
        ----------
        in_channel : int    
            channel of input tensor
        mid_channel : int
            channel of temporal variables
        num_classes : int
            number of classes
        """
        super(PAM, self).__init__()
        self.in_channel = in_channel
        self.out_channel = num_classes
        self.in_dim, self.out_dim = in_channel, in_channel
        self.key_dim, self.query_dim, self.value_dim = (mid_channel,
            mid_channel, in_channel)
        self.f_key = nn.Conv2d(self.in_dim, self.key_dim, 1)
        self.f_query = nn.Conv2d(self.in_dim, self.query_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.layer_idx = layer_idx
        self.branch_idx = branch_idx
        self.add_module(f'fc{layer_idx}_{branch_idx}', nn.Conv2d(self.
            in_dim, num_classes, 1))

    def forward(self, x):
        batch_size, _h, _w = x.size(0), x.size(2), x.size(3)
        value = x.view(batch_size, self.value_dim, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.query_dim, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_dim, -1)
        sim_map = torch.matmul(query, key)
        sim_map = self.key_dim ** -0.5 * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_dim, *x.size()[2:])
        fuse = self.gamma * context + x
        score = getattr(self, f'fc{self.layer_idx}_{self.branch_idx}')(fuse)
        return score


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'mid_channel': 4, 'num_classes': 4,
        'layer_idx': 1, 'branch_idx': 4}]
