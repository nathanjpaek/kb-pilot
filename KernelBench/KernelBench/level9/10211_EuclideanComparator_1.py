import torch
from dataclasses import dataclass
from collections import defaultdict
import torch.optim
from torch import nn


class Base(nn.Module):
    registered = defaultdict(dict)


    @dataclass
    class Config:
        pass

    @property
    def config(self):
        return self._config

    def __init__(self, *args, config: Config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config

    def __str__(self) ->str:
        return self.__name__

    @classmethod
    def module(Child, Impl):
        try:
            Impl.name
        except AttributeError:
            msg = 'Class {Impl} has no attribute .name'
            raise irtm.IRTMError(msg)
        Base.registered[Child.__name__][Impl.name] = Impl
        return Impl

    @classmethod
    def init(Child, *, name: str=None, **kwargs):
        try:
            if name is None:
                name = 'noop'
            A = Base.registered[Child.__name__][name]
        except KeyError:
            dicrep = yaml.dump(Base.registered, default_flow_style=False)
            msg = (
                f'could not find module "{name}"\n\navailable modules:\n{dicrep}'
                )
            raise irtm.IRTMError(msg)
        config = A.Config(**kwargs)
        log.info(f'! initializing {A.__name__} with {config}')
        return A(config=config)


class Comparator(Base):
    pass


@Comparator.module
class EuclideanComparator_1(Comparator):
    name = 'euclidean 1'

    def forward(self, X, Y):
        return torch.dist(X, Y, p=2) / X.shape[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
