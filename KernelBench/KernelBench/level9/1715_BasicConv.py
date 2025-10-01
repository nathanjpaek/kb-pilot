import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def singleton(cls):
    obj = cls()
    cls.__new__ = staticmethod(lambda cls: obj)
    try:
        del cls.__init__
    except AttributeError:
        pass
    return cls


@singleton
class model_debugger:

    def __init__(self):
        self.reset()

    def reset(self, filename_suffix=''):
        """
		Reset all the variables

		Parameters
		----------
		filename_suffix : str, optional
			the filename suffix, by default ""

		Returns
		-------
		model_debugger
			returns self
		"""
        self.model_forward_data = {}
        self.active_forwards = {}
        self.logs = []
        self.time_usage = {}
        self.active_timers = {}
        self.called_times = {}
        self.variable_log = {}
        self.tenserboard_writer = SummaryWriter(filename_suffix)
        self.epoch_write_counter = {}
        return self

    def write_to_tensorboard(self, name, value, epoch):
        """
		Add value to SummaryWriter

		Parameters
		----------
		name : str
			tag name
		value : number
			numeric value
		epoch : epoch
			Current epoch, if none a built in counter will be used
		"""
        if epoch is None:
            if name not in self.epoch_write_counter:
                self.epoch_write_counter[name] = 0
            epoch = self.epoch_write_counter[name]
            self.epoch_write_counter[name] += 1
        self.tenserboard_writer.add_scalar(name, value, epoch)

    def save_tensorboard(self):
        self.tenserboard_writer.close()

    def get_last_round(self):
        """
		Get the latest forward data for model

		Returns
		-------
		list
			with tuple of key value
		"""
        return self.model_forward_data.items()

    def start_forward(self, name):
        """
		Start a forward of component

		Parameters
		----------
		name : str
			name of the component
		"""
        if name not in self.active_forwards:
            self.active_forwards[name] = []

    def add_element(self, name, element):
        """
		Add a element to the forward path of a given name

		Parameters
		----------
		name : str
			name of component
		element : tensor/any
			the tensor of a given state
		"""
        self.active_forwards[name].append(element)

    def stop_forward(self, name):
        """
		Complete a forward

		Parameters
		----------
		name : str
			name of the component
		"""
        self.model_forward_data[name] = self.active_forwards[name]
        self.active_forwards[name] = []

    def log(self, value):
        """
		Log a value to list

		Parameters
		----------
		value : str
			log value
		"""
        self.logs.append(value)

    def start_track_time(self, name):
        """
		Track time usage of a given component
		- need to be closed by stop_track_time

		Parameters
		----------
		name : str
			name of timer
		"""
        if name not in self.active_timers:
            self.active_timers[name] = []
            self.time_usage[name] = 0
            self.called_times[name] = 0
        self.active_timers[name].append(time.time())
        self.called_times[name] += 1

    def stop_track_time(self, name):
        """
		Stops the timer with a given name

		Parameters
		----------
		name : str
			name of timer
		"""
        time_usage = time.time() - self.active_timers[name].pop(0)
        self.time_usage[name] += time_usage


class shared_backbone:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else
            'cpu')
        self.debugger = model_debugger()
        self.main_hidden_layer_size = 64


class BasicConv(nn.Module, shared_backbone):

    def __init__(self, output=120):
        super().__init__()
        shared_backbone.__init__(self)
        self.output_size = 96
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.fc1 = nn.Linear(self.output_size, output)

    def forward(self, x):
        """
		Basic forward operation on conv layer

		Parameters
		----------
		x : torch.tensor
			input data

		Returns
		-------
		torch.tensor
			the output after the operations
		"""
        if len(x.shape) < 4:
            padding = (1,) * (4 - len(x.shape))
            x = x.reshape(padding + x.shape)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = x.view(-1, self.output_size)
        assert x.shape[0] == 1
        assert len(x.shape) == 2
        return F.relu(self.fc1(x))


def get_inputs():
    return [torch.rand([4, 1, 6, 6])]


def get_init_inputs():
    return [[], {}]
