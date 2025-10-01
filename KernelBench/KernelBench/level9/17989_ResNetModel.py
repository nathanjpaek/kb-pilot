import torch
from typing import Dict
from abc import abstractmethod
from torch import nn
import torch.nn.functional as F


class DetectionModel(nn.Module):
    """
    Base class describing any single object detection model
    """

    def __init__(self, params: '{}'):
        self._params = params
        assert params is not None
        super(DetectionModel, self).__init__()

    def save(self, folder: 'Path', name):
        with open(folder / (name + '.json'), 'w') as f:
            json.dump(self._params, f, indent=2)
        torch.save(self.state_dict(), folder / (name + '.pt'))

    def load(self, folder: 'Path', name):
        with open(folder / (name + '.json'), 'r') as f:
            params = json.load(f)
        model = self.load_structure(params)
        model.load_state_dict(torch.load(folder / (name + '.pt')))
        return model

    @abstractmethod
    def load_structure(self, config_dictionary: 'Dict'):
        """
        Loads model structure from a json description.
        The structure json file describes non-trainable hyperparameters such as number of layers, filters etc.
        :param config_dictionary: dictionary with all the parameters necessary to load the model
        :return: None
        """
        return NotImplemented


class ResNetModel(DetectionModel):
    default_params = {'model_type': 'resnet', 'model_version': 1.0,
        'input_shape': (1, 240, 320), 'initial_filters': 16, 'num_outputs': 2}

    def __init__(self, params=None):
        self._model_version = 1.0
        if params is None:
            params = ResNetModel.default_params
        if 'version' not in params:
            params['version'] = self._model_version
        if 'model_type' not in params:
            params['model_type'] = 'resnet'
        super(ResNetModel, self).__init__(params)
        self.load_structure(params)

    def load_structure(self, hyperparameters):
        C_in, _H_in, _W_in = hyperparameters['input_shape']
        init_f = hyperparameters['initial_filters']
        num_outputs = hyperparameters['num_outputs']
        self.conv1 = nn.Conv2d(C_in, init_f, kernel_size=3, stride=2, padding=1
            )
        self.conv2 = nn.Conv2d(init_f + C_in, 2 * init_f, kernel_size=3,
            stride=1, padding=1)
        self.conv3 = nn.Conv2d(3 * init_f + C_in, 4 * init_f, kernel_size=3,
            padding=1)
        self.conv4 = nn.Conv2d(7 * init_f + C_in, 8 * init_f, kernel_size=3,
            padding=1)
        self.conv5 = nn.Conv2d(15 * init_f + C_in, 16 * init_f, kernel_size
            =3, padding=1)
        self.fc1 = nn.Linear(16 * init_f, 64 * init_f)
        self.fc_out = nn.Linear(64 * init_f, num_outputs)
        return self

    def forward(self, x):
        identity = F.avg_pool2d(x, 4, 4)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)
        identity = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)
        identity = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)
        identity = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)
        x = F.relu(self.conv5(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc_out(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
