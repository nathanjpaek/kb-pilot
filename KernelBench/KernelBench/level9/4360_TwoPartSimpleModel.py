import torch
import torch.nn as nn
import torch.utils.data


class SimpleModel(nn.Module):

    def forward(self, x):
        return 2 * x

    def prepare_for_export(self, cfg, inputs, predictor_type):
        return PredictorExportConfig(model=self, data_generator=lambda x: (x,))


class TwoPartSimpleModel(nn.Module):
    """
    Suppose there're some function in the middle that can't be traced, therefore we
    need to export the model as two parts.
    """

    def __init__(self):
        super().__init__()
        self.part1 = SimpleModel()
        self.part2 = SimpleModel()

    def forward(self, x):
        x = self.part1(x)
        x = TwoPartSimpleModel.non_traceable_func(x)
        x = self.part2(x)
        return x

    def prepare_for_export(self, cfg, inputs, predictor_type):

        def data_generator(x):
            part1_args = x,
            x = self.part1(x)
            x = TwoPartSimpleModel.non_traceable_func(x)
            part2_args = x,
            return {'part1': part1_args, 'part2': part2_args}
        return PredictorExportConfig(model={'part1': self.part1, 'part2':
            self.part2}, data_generator=data_generator, run_func_info=
            FuncInfo.gen_func_info(TwoPartSimpleModel.RunFunc, params={}))

    @staticmethod
    def non_traceable_func(x):
        return x + 1 if len(x.shape) > 3 else x - 1


    class RunFunc(object):

        def __call__(self, model, x):
            assert isinstance(model, dict)
            x = model['part1'](x)
            x = TwoPartSimpleModel.non_traceable_func(x)
            x = model['part2'](x)
            return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
