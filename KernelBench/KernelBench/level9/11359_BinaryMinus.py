import abc
import inspect
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from typing import Any
from typing import *


def get_module_name(cls_or_func):
    module_name = cls_or_func.__module__
    if module_name == '__main__':
        for frm in inspect.stack():
            if inspect.getmodule(frm[0]).__name__ == '__main__':
                main_file_path = Path(inspect.getsourcefile(frm[0]))
                if not Path().samefile(main_file_path.parent):
                    raise RuntimeError(
                        f'You are using "{main_file_path}" to launch your experiment, please launch the experiment under the directory where "{main_file_path.name}" is located.'
                        )
                module_name = main_file_path.stem
                break
    if module_name == '__main__':
        warnings.warn(
            'Callstack exhausted but main module still not found. This will probably cause issues that the function/class cannot be imported.'
            )
    if (f'{cls_or_func.__module__}.{cls_or_func.__name__}' ==
        'torch.nn.modules.rnn.LSTM'):
        module_name = cls_or_func.__module__
    return module_name


def reset_uid(namespace: 'str'='default') ->None:
    _last_uid[namespace] = 0


def _create_wrapper_cls(cls, store_init_parameters=True, reset_mutation_uid
    =False, stop_parsing=True):


    class wrapper(cls):

        def __init__(self, *args, **kwargs):
            self._stop_parsing = stop_parsing
            if reset_mutation_uid:
                reset_uid('mutation')
            if store_init_parameters:
                argname_list = list(inspect.signature(cls.__init__).
                    parameters.keys())[1:]
                full_args = {}
                full_args.update(kwargs)
                assert len(args) <= len(argname_list
                    ), f'Length of {args} is greater than length of {argname_list}.'
                for argname, value in zip(argname_list, args):
                    full_args[argname] = value
                args = list(args)
                for i, value in enumerate(args):
                    if isinstance(value, Translatable):
                        args[i] = value._translate()
                for i, value in kwargs.items():
                    if isinstance(value, Translatable):
                        kwargs[i] = value._translate()
                self._init_parameters = full_args
            else:
                self._init_parameters = {}
            super().__init__(*args, **kwargs)
    wrapper.__module__ = get_module_name(cls)
    wrapper.__name__ = cls.__name__
    wrapper.__qualname__ = cls.__qualname__
    wrapper.__init__.__doc__ = cls.__init__.__doc__
    return wrapper


def serialize_cls(cls):
    """
    To create an serializable class.
    """
    return _create_wrapper_cls(cls)


def basic_unit(cls):
    """
    To wrap a module as a basic unit, to stop it from parsing and make it mutate-able.
    """
    import torch.nn as nn
    assert issubclass(cls, nn.Module
        ), 'When using @basic_unit, the class must be a subclass of nn.Module.'
    return serialize_cls(cls)


class Translatable(abc.ABC):
    """
    Inherit this class and implement ``translate`` when the inner class needs a different
    parameter from the wrapper class in its init function.
    """

    @abc.abstractmethod
    def _translate(self) ->Any:
        pass


@basic_unit
class BinaryMinus(nn.Module):

    def forward(self, x):
        return x[0] - x[1]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
