import torch
import numpy as np
import torch.optim
import torch.nn as nn
import torch.nn.utils
import torch.autograd


class BaseMetric:
    """ Base class for all the metrics """

    def __init__(self, name):
        self.name = name

    def calculate(self, batch_info):
        """ Calculate value of a metric based on supplied data """
        raise NotImplementedError

    def reset(self):
        """ Reset value of a metric """
        raise NotImplementedError

    def value(self):
        """ Return current value for the metric """
        raise NotImplementedError

    def write_state_dict(self, training_info: 'TrainingInfo',
        hidden_state_dict: 'dict') ->None:
        """ Potentially store some metric state to the checkpoint """
        pass

    def load_state_dict(self, training_info: 'TrainingInfo',
        hidden_state_dict: 'dict') ->None:
        """ Potentially load some metric state from the checkpoint """
        pass


class AveragingMetric(BaseMetric):
    """ Base class for metrics that simply calculate the average over the epoch """

    def __init__(self, name):
        super().__init__(name)
        self.storage = []

    def calculate(self, batch_info):
        """ Calculate value of a metric """
        value = self._value_function(batch_info)
        self.storage.append(value)

    def _value_function(self, batch_info):
        raise NotImplementedError

    def reset(self):
        """ Reset value of a metric """
        self.storage = []

    def value(self):
        """ Return current value for the metric """
        return float(np.mean(self.storage))


class Loss(AveragingMetric):
    """ Just a loss function """

    def __init__(self):
        super().__init__('loss')

    def _value_function(self, batch_info):
        """ Just forward a value of the loss"""
        return batch_info['loss'].item()


class Model(nn.Module):
    """ Class representing full neural network model """

    def metrics(self) ->list:
        """ Set of metrics for this model """
        return [Loss()]

    def train(self, mode=True):
        """
        Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Returns:
            Module: self
        """
        super().train(mode)
        if mode:
            mu.apply_leaf(self, mu.set_train_mode)
        return self

    def summary(self, input_size=None, hashsummary=False):
        """ Print a model summary """
        if input_size is None:
            None
            None
            sum(p.numel() for p in self.model.parameters())
            None
            None
        else:
            summary(self, input_size)
        if hashsummary:
            for idx, hashvalue in enumerate(self.hashsummary()):
                None

    def hashsummary(self):
        """ Print a model summary - checksums of each layer parameters """
        children = list(self.children())
        result = []
        for child in children:
            result.extend(hashlib.sha256(x.detach().cpu().numpy().tobytes()
                ).hexdigest() for x in child.parameters())
        return result

    def get_layer_groups(self):
        """ Return layers grouped """
        return [self]

    def reset_weights(self):
        """ Call proper initializers for the weights """
        pass

    @property
    def is_recurrent(self) ->bool:
        """ If the network is recurrent and needs to be fed state as well as the observations """
        return False


class BackboneModel(Model):
    """ Model that serves as a backbone network to connect your heads to """


class ImageToTensor(BackboneModel):
    """
    Convert simple image to tensor.

    Flip channels to a [C, W, H] order and potentially convert 8-bit color values to floats
    """

    def __init__(self):
        super().__init__()

    def reset_weights(self):
        pass

    def forward(self, image):
        result = image.permute(0, 3, 1, 2).contiguous()
        if result.dtype == torch.uint8:
            result = result.type(torch.float) / 255.0
        else:
            result = result.type(torch.float)
        return result


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
