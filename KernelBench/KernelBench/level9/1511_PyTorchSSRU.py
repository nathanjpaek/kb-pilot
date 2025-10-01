import torch
from typing import Tuple
from abc import abstractmethod
import torch as pt
import torch.distributed
import torch.distributed.elastic.multiprocessing.errors


class AutoregressiveLayer(pt.nn.Module):

    @property
    @abstractmethod
    def num_state_tensors(self) ->int:
        """ Number of state tensors returned by the layer """
        raise NotImplementedError

    @property
    @abstractmethod
    def needs_mask(self) ->bool:
        """ Whether the layer makes use of a mask tensor or not """
        raise NotImplementedError

    @abstractmethod
    def get_state_shape(self, batch_size) ->Tuple:
        """
        :param batch_size: current batch size
        :return: dimensions of each output state (assuming all of them have the same shape)
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: 'pt.Tensor', previous_states: 'pt.Tensor', *args
        ) ->Tuple:
        """
        :param inputs: layer input
        :param previous_states: Previous states array or list of arrays
        :param args: layer-specific arguments and/or arguments to be ignored
        :return: layer output and new states
        """
        raise NotImplementedError


class PyTorchSSRU(AutoregressiveLayer):
    """
    Simpler Simple Recurrent Unit

    Kim et al, "From Research to Production and Back: Ludicrously Fast Neural Machine Translation" WNGT 2019

    Variant of an LSTM cell aimed at reducing computational dependency across time steps.
    Formally described as:

    (1) f[t] = sigmoid(W1[t] * x[t] + b[t])
    (2) c[t] = f[t] . c[t-1] + (1 - f[t]) . W2[t] * x[t]
    (3) h = ReLU(c[t])

    where:
        . represents elementwise multiplication;
        x[t] is the input at time step t;
        f[t] is the output of the forget gate at time step t;
        c[t] is the cell state at time step t;
        h is the output of the unit.

    :param model_size: number of hidden units
    :param inference_only: flag used to indicate execution at inference time
    """

    def __init__(self, model_size: 'int', inference_only: 'bool') ->None:
        super().__init__()
        self.model_size = model_size
        self.inference_only = inference_only
        self.cell_state_transform = (self._inference_cell_state_transform if
            inference_only else self._training_cell_state_transform)
        self.forget_gate = pt.nn.Linear(in_features=model_size,
            out_features=model_size, bias=True)
        self.forget_gate_act = pt.nn.Sigmoid()
        self.linear = pt.nn.Linear(in_features=model_size, out_features=
            model_size, bias=False)
        self.relu = pt.nn.ReLU(inplace=False)

    @property
    def num_state_tensors(self) ->int:
        """ Number of state tensors returned by the layer """
        return 1

    @property
    def needs_mask(self) ->bool:
        """ Whether the layer makes use of a mask tensor or not """
        return False

    def get_state_shape(self, batch_size: 'int') ->Tuple:
        """
        :param batch_size: current batch size
        :return: dimensions of each output state (assuming all of them have the same shape)
        """
        return 1, batch_size, self.model_size

    @staticmethod
    @pt.jit.script_if_tracing
    def _training_cell_state_transform(previous_cell_state, weighted_inputs,
        forget_rates) ->Tuple[pt.Tensor, pt.Tensor]:
        """Update SSRU cell at training time"""
        steps = weighted_inputs.size()[0]
        cell_state = previous_cell_state.squeeze(0)
        states = []
        for t in range(steps):
            cell_state = forget_rates[t, :, :] * cell_state + weighted_inputs[
                t, :, :]
            states.append(cell_state)
        states = pt.stack(states, dim=0)
        return states, cell_state.unsqueeze(0)

    @staticmethod
    def _inference_cell_state_transform(previous_cell_state,
        weighted_inputs, forget_rates) ->Tuple[pt.Tensor, pt.Tensor]:
        """Update SSRU cell at inference time"""
        new_step_state = forget_rates * previous_cell_state + weighted_inputs
        return new_step_state, new_step_state

    def forward(self, inputs: 'pt.Tensor', previous_states: 'pt.Tensor', **args
        ) ->Tuple[pt.Tensor, pt.Tensor]:
        """
        :param inputs: input data. Shape: (max_length, batch, input_depth).
        :param previous_states: previous cell states. Shape: (max_length, batch, input_depth)
        :return: cell output and new cell states.  Both with shape (max_length, batch, input_depth).
        """
        forget_rates = self.forget_gate_act(self.forget_gate(inputs))
        weighted_inputs = (1 - forget_rates) * self.linear(inputs)
        cell_state, last_step_state = self.cell_state_transform(previous_states
            , weighted_inputs, forget_rates)
        return self.relu(cell_state), last_step_state

    def weights_from_mxnet_block(self, block_mx: "'SSRU'"):
        self.forget_gate.weight.data[:] = pt.as_tensor(block_mx.forget_gate
            .weight.data().asnumpy())
        self.forget_gate.bias.data[:] = pt.as_tensor(block_mx.forget_gate.
            bias.data().asnumpy())
        self.linear.weight.data[:] = pt.as_tensor(block_mx.linear.weight.
            data().asnumpy())


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'model_size': 4, 'inference_only': 4}]
